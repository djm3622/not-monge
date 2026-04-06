"""Modern OT baselines: OTP-style minimax and flow-based OT."""

from __future__ import annotations

from typing import Any, Mapping

import numpy as np
import ot
import torch
import torch.nn.functional as F

from src.models.flow import build_vector_field
from src.models.ot_map import build_ot_map
from src.models.potential import build_potential
from src.solvers.base import BaseOTSolver
from src.solvers.minimax_ot import MinimaxOTSolver, frozen_parameters
from src.solvers.registry import register_solver
from src.training.losses import (
    minimax_potential_objective,
    minimax_transport_objective,
    quadratic_cost,
)
from src.utils.ode import integrate_ode


def _uniform_weights(num_points: int) -> tuple[np.ndarray, np.ndarray]:
    weights = np.full(num_points, 1.0 / num_points, dtype=np.float64)
    return weights, weights.copy()


def _sinkhorn_plan(source: torch.Tensor, target: torch.Tensor, reg: float) -> torch.Tensor:
    """Compute a batchwise entropic transport plan."""
    a, b = _uniform_weights(source.shape[0])
    cost_matrix = torch.cdist(source.detach(), target.detach()).pow(2).cpu().numpy()
    plan = ot.sinkhorn(a, b, cost_matrix, reg=reg, method="sinkhorn_log", numItermax=5000)
    return torch.from_numpy(np.asarray(plan, dtype=np.float32)).to(source.device)


def _barycentric_projection(plan: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    row_sums = plan.sum(dim=1, keepdim=True).clamp_min(1.0e-8)
    return plan @ target / row_sums


def _negative_entropy(plan: torch.Tensor) -> torch.Tensor:
    safe_plan = plan.clamp_min(1.0e-8)
    return (safe_plan * safe_plan.log()).sum()


def _rbf_mmd_loss(samples_a: torch.Tensor, samples_b: torch.Tensor) -> torch.Tensor:
    combined = torch.cat([samples_a, samples_b], dim=0)
    pairwise = torch.cdist(combined, combined).pow(2)
    bandwidth = pairwise.detach().median().clamp_min(1.0e-6)
    gamma = 1.0 / (2.0 * bandwidth)
    kernel = torch.exp(-gamma * pairwise)
    n = samples_a.shape[0]
    xx = kernel[:n, :n].mean()
    yy = kernel[n:, n:].mean()
    xy = kernel[:n, n:].mean()
    return xx + yy - 2.0 * xy


def _potential_gradient_penalty(
    potential: torch.nn.Module,
    real_samples: torch.Tensor,
    fake_samples: torch.Tensor,
) -> torch.Tensor:
    alpha = torch.rand(real_samples.shape[0], 1, device=real_samples.device, dtype=real_samples.dtype)
    interpolated = alpha * real_samples + (1.0 - alpha) * fake_samples
    interpolated.requires_grad_(True)
    with torch.enable_grad():
        values = potential(interpolated)
        gradients = torch.autograd.grad(values.sum(), interpolated, create_graph=True)[0]
    return (gradients.norm(dim=-1) - 1.0).pow(2).mean()


class OTPMinimaxSolver(MinimaxOTSolver):
    """Stabilized semi-dual minimax OT with smoothing and plan supervision."""

    solver_name = "otp"
    solver_group = "learned_w2"
    supports_training = True
    supports_potential = True

    def __init__(
        self,
        map_config: Mapping[str, Any],
        solver_config: Mapping[str, Any],
        training_config: Mapping[str, Any],
    ) -> None:
        super().__init__(map_config=map_config, solver_config=solver_config, training_config=training_config)
        # Preserve the attribute names used throughout the OTP implementation after
        # the shared minimax solver was refactored to forward_/inverse_ naming.
        self.potential = self.forward_potential
        self.transport = self.forward_potential
        self.critic_steps = self.inverse_steps
        smoothing_cfg = dict(solver_config.get("smoothing", {}))
        plan_cfg = dict(solver_config.get("plan", {}))
        regularization_cfg = dict(solver_config.get("regularization", {}))
        self.sigma_start = float(smoothing_cfg.get("sigma_start", 0.05))
        self.sigma_end = float(smoothing_cfg.get("sigma_end", 0.0))
        self.anneal_steps = int(smoothing_cfg.get("anneal_steps", 1000))
        self.plan_enabled = bool(plan_cfg.get("enabled", True))
        self.plan_reg = float(plan_cfg.get("reg", 0.1))
        self.plan_supervision_weight = float(plan_cfg.get("supervision_weight", 0.5))
        self.plan_entropy_weight = float(plan_cfg.get("entropy_weight", 0.0))
        self.potential_gp_weight = float(regularization_cfg.get("potential_gp_weight", 1.0))
        self.potential_l2_weight = float(regularization_cfg.get("potential_l2_weight", 1.0e-3))
        self.train_step_index = 0

    def _smoothing_sigma(self) -> float:
        progress = min(self.train_step_index / max(self.anneal_steps, 1), 1.0)
        return self.sigma_start + (self.sigma_end - self.sigma_start) * progress

    def _apply_smoothing(self, batch: Mapping[str, torch.Tensor], sigma: float) -> dict[str, torch.Tensor]:
        source = batch["source"]
        target = batch["target"]
        if sigma <= 0.0:
            return {
                "source": source,
                "target": target,
                "ground_truth_map": batch["ground_truth_map"],
            }
        return {
            "source": source + sigma * torch.randn_like(source),
            "target": target + sigma * torch.randn_like(target),
            "ground_truth_map": batch["ground_truth_map"],
        }

    def compute_loss(self, batch: Mapping[str, torch.Tensor], smoothing_sigma: float) -> dict[str, torch.Tensor]:
        smoothed_batch = self._apply_smoothing(batch, smoothing_sigma)
        source = smoothed_batch["source"]
        target = smoothed_batch["target"]
        transported = self.compute_map(source)
        potential_real = self.potential(target)
        potential_fake = self.potential(transported)
        critic_objective = minimax_potential_objective(potential_real, potential_fake)
        map_objective = minimax_transport_objective(source, transported, potential_fake)
        if self.plan_enabled:
            plan = _sinkhorn_plan(transported, target, reg=self.plan_reg)
        else:
            plan = torch.full(
                (source.shape[0], target.shape[0]),
                1.0 / (source.shape[0] * target.shape[0]),
                device=source.device,
                dtype=source.dtype,
            )
        barycentric_target = _barycentric_projection(plan.detach(), target)
        return {
            "critic_objective": critic_objective,
            "map_objective": map_objective,
            "transported": transported,
            "potential_real": potential_real,
            "potential_fake": potential_fake,
            "plan": plan,
            "barycentric_target": barycentric_target,
            "source": source,
            "target": target,
        }

    def regularization(
        self,
        batch: Mapping[str, torch.Tensor],
        transported: torch.Tensor,
        potential_real: torch.Tensor,
        potential_fake: torch.Tensor,
        plan: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        del potential_fake
        barycentric_target = _barycentric_projection(plan.detach(), batch["target"])
        return {
            "potential_gp": _potential_gradient_penalty(
                self.potential,
                batch["target"],
                transported.detach(),
            ),
            "potential_l2": potential_real.pow(2).mean(),
            "plan_supervision": F.mse_loss(transported, barycentric_target),
            "negative_entropy": _negative_entropy(plan),
        }

    def state_dict(self, *args: Any, **kwargs: Any) -> dict[str, Any]:  # type: ignore[override]
        state = super().state_dict(*args, **kwargs)
        state["train_step_index"] = self.train_step_index
        return state

    def load_state_dict(self, state_dict: Mapping[str, Any], strict: bool = True) -> None:  # type: ignore[override]
        self.train_step_index = int(state_dict.get("train_step_index", 0))
        inherited_state = {key: value for key, value in state_dict.items() if key != "train_step_index"}
        super().load_state_dict(inherited_state, strict=strict)

    def training_step(
        self,
        batch: Mapping[str, torch.Tensor],
        scaler: torch.amp.GradScaler,
        autocast_context: Any,
        gradient_clip_norm: float | None,
    ) -> dict[str, float]:
        sigma = self._smoothing_sigma()
        critic_values = []
        gp_values = []
        l2_values = []
        for _ in range(self.critic_steps):
            potential_optimizer = self.optimizers[1]
            potential_optimizer.zero_grad(set_to_none=True)
            with autocast_context():
                loss_terms = self.compute_loss(batch, sigma)
                regs = self.regularization(
                    batch={"source": loss_terms["source"], "target": loss_terms["target"]},
                    transported=loss_terms["transported"],
                    potential_real=loss_terms["potential_real"],
                    potential_fake=loss_terms["potential_fake"],
                    plan=loss_terms["plan"],
                )
                stabilized_objective = (
                    loss_terms["critic_objective"]
                    - self.potential_gp_weight * regs["potential_gp"]
                    - self.potential_l2_weight * regs["potential_l2"]
                )
                critic_loss = -stabilized_objective
            scaler.scale(critic_loss).backward()
            if gradient_clip_norm is not None:
                scaler.unscale_(potential_optimizer)
                torch.nn.utils.clip_grad_norm_(self.potential.parameters(), gradient_clip_norm)
            scaler.step(potential_optimizer)
            critic_values.append(float(loss_terms["critic_objective"].detach()))
            gp_values.append(float(regs["potential_gp"].detach()))
            l2_values.append(float(regs["potential_l2"].detach()))

        map_optimizer = self.optimizers[0]
        map_optimizer.zero_grad(set_to_none=True)
        with frozen_parameters(self.potential):
            with autocast_context():
                loss_terms = self.compute_loss(batch, sigma)
                regs = self.regularization(
                    batch={"source": loss_terms["source"], "target": loss_terms["target"]},
                    transported=loss_terms["transported"],
                    potential_real=loss_terms["potential_real"],
                    potential_fake=loss_terms["potential_fake"],
                    plan=loss_terms["plan"],
                )
                map_loss = (
                    loss_terms["map_objective"]
                    + self.plan_supervision_weight * regs["plan_supervision"]
                    + self.plan_entropy_weight * regs["negative_entropy"]
                )
                cost = quadratic_cost(loss_terms["source"], loss_terms["transported"]).mean()
        scaler.scale(map_loss).backward()
        if gradient_clip_norm is not None:
            scaler.unscale_(map_optimizer)
            torch.nn.utils.clip_grad_norm_(self.transport.parameters(), gradient_clip_norm)
        scaler.step(map_optimizer)
        scaler.update()
        for scheduler in self.schedulers:
            scheduler.step()
        self.train_step_index += 1
        return {
            "train/map_loss": float(map_loss.detach()),
            "train/cost": float(cost.detach()),
            "train/map_l2": float((loss_terms["transported"].detach() - batch["target"]).pow(2).mean().sqrt()),
            "train/critic_objective": sum(critic_values) / len(critic_values),
            "train/potential_gp": sum(gp_values) / len(gp_values),
            "train/potential_l2": sum(l2_values) / len(l2_values),
            "train/plan_supervision": float(regs["plan_supervision"].detach()),
            "train/smoothing_sigma": sigma,
        }


class FlowOTSolver(BaseOTSolver):
    """Deterministic neural-ODE style transport solver."""

    solver_name = "flow"
    solver_group = "advanced_alt"
    supports_training = True
    supports_potential = False

    def __init__(
        self,
        map_config: Mapping[str, Any],
        solver_config: Mapping[str, Any],
        training_config: Mapping[str, Any],
    ) -> None:
        super().__init__(training_config=training_config)
        velocity_cfg = dict(solver_config.get("velocity", {}))
        velocity_cfg["input_dim"] = int(map_config["input_dim"])
        self.velocity = build_vector_field(velocity_cfg)
        integration_cfg = dict(solver_config.get("integration", {}))
        plan_cfg = dict(solver_config.get("plan", {}))
        loss_cfg = dict(solver_config.get("loss", {}))
        self.learning_rate = float(solver_config.get("lr", training_config["optimizer"]["lr"]))
        self.integration_backend = str(integration_cfg.get("backend", "auto"))
        self.integration_method = str(integration_cfg.get("method", "rk4"))
        self.integration_steps = int(integration_cfg.get("steps", 8))
        self.integration_atol = float(integration_cfg.get("atol", 1.0e-5))
        self.integration_rtol = float(integration_cfg.get("rtol", 1.0e-5))
        self.use_adjoint = bool(integration_cfg.get("use_adjoint", False))
        self.plan_reg = float(plan_cfg.get("reg", 0.1))
        self.endpoint_weight = float(loss_cfg.get("endpoint_weight", 1.0))
        self.energy_weight = float(loss_cfg.get("energy_weight", 0.1))
        self.mmd_weight = float(loss_cfg.get("mmd_weight", 0.0))

    def forward_flow(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        return self.velocity(x, t)

    def integrate_flow(
        self,
        x: torch.Tensor,
        return_trajectory: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        return integrate_ode(
            func=self.forward_flow,
            initial_state=x,
            steps=self.integration_steps,
            backend=self.integration_backend,
            method=self.integration_method,
            atol=self.integration_atol,
            rtol=self.integration_rtol,
            use_adjoint=self.use_adjoint,
            return_trajectory=return_trajectory,
        )

    def compute_map(self, x: torch.Tensor) -> torch.Tensor:
        result = self.integrate_flow(x, return_trajectory=False)
        assert torch.is_tensor(result)
        return result

    def configure_optimizers(self, total_steps: int) -> None:
        self._configure_multi_optimizer(
            parameter_groups=[(self.velocity.parameters(), self.learning_rate)],
            total_steps=total_steps,
        )

    def state_dict(self, *args: Any, **kwargs: Any) -> dict[str, Any]:  # type: ignore[override]
        return {
            "velocity": self.velocity.state_dict(*args, **kwargs),
            "optimizers": [optimizer.state_dict() for optimizer in self.optimizers],
            "schedulers": [scheduler.state_dict() for scheduler in self.schedulers],
        }

    def load_state_dict(self, state_dict: Mapping[str, Any], strict: bool = True) -> None:  # type: ignore[override]
        self.velocity.load_state_dict(state_dict["velocity"], strict=strict)
        for optimizer, optimizer_state in zip(self.optimizers, state_dict.get("optimizers", [])):
            optimizer.load_state_dict(optimizer_state)
        for scheduler, scheduler_state in zip(self.schedulers, state_dict.get("schedulers", [])):
            scheduler.load_state_dict(scheduler_state)

    def _path_energy(self, trajectory: torch.Tensor) -> torch.Tensor:
        times = torch.linspace(0.0, 1.0, trajectory.shape[0], device=trajectory.device, dtype=trajectory.dtype)
        dt = 1.0 / max(trajectory.shape[0] - 1, 1)
        energy = torch.zeros((), device=trajectory.device, dtype=trajectory.dtype)
        for index, time in enumerate(times[:-1]):
            velocity = self.forward_flow(trajectory[index], time.expand(trajectory[index].shape[0]))
            energy = energy + 0.5 * dt * velocity.pow(2).sum(dim=-1).mean()
        return energy

    def training_step(
        self,
        batch: Mapping[str, torch.Tensor],
        scaler: torch.amp.GradScaler,
        autocast_context: Any,
        gradient_clip_norm: float | None,
    ) -> Mapping[str, float]:
        optimizer = self.optimizers[0]
        optimizer.zero_grad(set_to_none=True)
        source = batch["source"]
        target = batch["target"]
        plan = _sinkhorn_plan(source, target, reg=self.plan_reg)
        barycentric_target = _barycentric_projection(plan.detach(), target)
        with autocast_context():
            integrated, trajectory = self.integrate_flow(source, return_trajectory=True)
            assert torch.is_tensor(integrated)
            assert torch.is_tensor(trajectory)
            endpoint_loss = F.mse_loss(integrated, barycentric_target)
            energy = self._path_energy(trajectory)
            terminal_mmd = _rbf_mmd_loss(integrated, target)
            loss = (
                self.endpoint_weight * endpoint_loss
                + self.energy_weight * energy
                + self.mmd_weight * terminal_mmd
            )
        scaler.scale(loss).backward()
        if gradient_clip_norm is not None:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(self.velocity.parameters(), gradient_clip_norm)
        scaler.step(optimizer)
        scaler.update()
        self.schedulers[0].step()
        return {
            "train/flow_loss": float(loss.detach()),
            "train/endpoint_loss": float(endpoint_loss.detach()),
            "train/path_energy": float(energy.detach()),
            "train/terminal_mmd": float(terminal_mmd.detach()),
            "train/map_l2": float((integrated.detach() - batch["target"]).pow(2).mean().sqrt()),
        }


@register_solver("otp")
def _build_otp_solver(
    model_config: Mapping[str, Any],
    solver_config: Mapping[str, Any],
    training_config: Mapping[str, Any],
) -> OTPMinimaxSolver:
    return OTPMinimaxSolver(model_config, solver_config, training_config)


@register_solver("flow")
def _build_flow_solver(
    model_config: Mapping[str, Any],
    solver_config: Mapping[str, Any],
    training_config: Mapping[str, Any],
) -> FlowOTSolver:
    return FlowOTSolver(model_config, solver_config, training_config)
