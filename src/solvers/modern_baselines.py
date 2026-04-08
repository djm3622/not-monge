"""Modern OT baselines: OTP and flow-based OT."""

from __future__ import annotations

import math
from typing import Any, Mapping

import numpy as np
import ot
import torch
import torch.nn.functional as F

from src.models.flow import build_vector_field
from src.models.ot_map import build_ot_map
from src.models.potential import build_potential
from src.solvers.base import BaseOTSolver
from src.solvers.minimax_ot import frozen_parameters
from src.solvers.registry import register_solver
from src.training.losses import quadratic_cost
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


class OTPMinimaxSolver(BaseOTSolver):
    """Paper-faithful OTP solver with a smoothed source measure."""

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
        super().__init__(training_config=training_config)
        input_dim = int(map_config["input_dim"])
        output_dim = int(map_config.get("output_dim", input_dim))

        transport_cfg = dict(solver_config.get("transport", map_config))
        transport_cfg["input_dim"] = input_dim
        transport_cfg["output_dim"] = output_dim
        self.transport = build_ot_map(transport_cfg)

        potential_cfg = dict(solver_config.get("potential", {}))
        potential_cfg["input_dim"] = output_dim
        potential_cfg.setdefault("kind", "denseicnn")
        potential_cfg.setdefault("activation", "celu")
        potential_cfg.setdefault("strong_convexity", 0.0)
        potential_cfg.setdefault("identity_quadratic", 0.0)
        self.potential_backbone = build_potential(potential_cfg)

        self.transport_lr = float(
            solver_config.get("transport_lr", solver_config.get("map_lr", training_config["optimizer"]["lr"]))
        )
        self.potential_lr = float(solver_config.get("potential_lr", training_config["optimizer"]["lr"]))
        self.transport_steps = int(solver_config.get("transport_steps", solver_config.get("inner_steps", 10)))

        optimizer_cfg = dict(solver_config.get("optimizer", {}))
        betas = optimizer_cfg.get("betas", (0.0, 0.9))
        self.optimizer_betas = (float(betas[0]), float(betas[1]))
        self.optimizer_weight_decay = float(optimizer_cfg.get("weight_decay", 0.0))
        self.gradient_clip_enabled = bool(solver_config.get("gradient_clip_enabled", False))

        self.use_c_concave_parameterization = bool(solver_config.get("use_c_concave_parameterization", True))
        self.quadratic_scale = float(solver_config.get("quadratic_scale", 0.5))

        noise_cfg = dict(solver_config.get("noise", solver_config.get("smoothing", {})))
        self.noise_kind = str(noise_cfg.get("kind", "additive_gaussian")).lower()
        self.noise_start = float(noise_cfg.get("sigma_start", noise_cfg.get("start", 0.2)))
        self.noise_end = float(noise_cfg.get("sigma_end", noise_cfg.get("end", 0.05)))
        self.noise_update_every = int(noise_cfg.get("update_every", 0))
        self.noise_anneal_steps = int(noise_cfg.get("anneal_steps", 0))

        self.train_step_index = 0
        self._total_steps = 1

    def _transport_cost(self, source: torch.Tensor, transported: torch.Tensor) -> torch.Tensor:
        return self.quadratic_scale * (source - transported).pow(2).sum(dim=-1)

    def _dual_potential(self, target: torch.Tensor) -> torch.Tensor:
        backbone = self.potential_backbone(target)
        if not self.use_c_concave_parameterization:
            return backbone
        quadratic = self.quadratic_scale * target.pow(2).sum(dim=-1, keepdim=True)
        return quadratic - backbone

    def _noise_progress(self) -> float:
        if self.noise_anneal_steps > 0:
            return min(self.train_step_index / max(self.noise_anneal_steps, 1), 1.0)
        interval = self.noise_update_every if self.noise_update_every > 0 else max(self._total_steps // 10, 1)
        scheduled_step = min((self.train_step_index // interval) * interval + 1, self._total_steps)
        return min(scheduled_step / max(self._total_steps, 1), 1.0)

    def current_noise_level(self) -> float:
        progress = self._noise_progress()
        if self.noise_kind in {"variance_preserving", "vp"}:
            t = 1.0 - progress
            return 1.0 - math.exp(-0.5 * (self.noise_start - self.noise_end) * t * t - self.noise_end * t)
        return self.noise_start + (self.noise_end - self.noise_start) * progress

    def _perturb_source(self, source: torch.Tensor) -> torch.Tensor:
        level = self.current_noise_level()
        if level <= 0.0:
            return source
        noise = torch.randn_like(source)
        if self.noise_kind in {"variance_preserving", "vp"}:
            return math.sqrt(max(1.0 - level, 0.0)) * source + math.sqrt(level) * noise
        return source + level * noise

    def compute_map(self, x: torch.Tensor) -> torch.Tensor:
        return self.transport(x)

    def compute_potential(self, x: torch.Tensor) -> torch.Tensor | None:
        return self._dual_potential(x)

    def configure_optimizers(self, total_steps: int) -> None:
        self._total_steps = max(int(total_steps), 1)
        self.optimizers = [
            torch.optim.Adam(
                self.transport.parameters(),
                lr=self.transport_lr,
                betas=self.optimizer_betas,
                weight_decay=self.optimizer_weight_decay,
            ),
            torch.optim.Adam(
                self.potential_backbone.parameters(),
                lr=self.potential_lr,
                betas=self.optimizer_betas,
                weight_decay=self.optimizer_weight_decay,
            ),
        ]
        self.schedulers = []

    def state_dict(self, *args: Any, **kwargs: Any) -> dict[str, Any]:  # type: ignore[override]
        return {
            "transport": self.transport.state_dict(*args, **kwargs),
            "potential_backbone": self.potential_backbone.state_dict(*args, **kwargs),
            "optimizers": [optimizer.state_dict() for optimizer in self.optimizers],
            "train_step_index": self.train_step_index,
            "total_steps": self._total_steps,
        }

    def load_state_dict(self, state_dict: Mapping[str, Any], strict: bool = True) -> None:  # type: ignore[override]
        self.train_step_index = int(state_dict.get("train_step_index", 0))
        self._total_steps = int(state_dict.get("total_steps", self._total_steps))
        self.transport.load_state_dict(state_dict["transport"], strict=strict)
        self.potential_backbone.load_state_dict(state_dict["potential_backbone"], strict=strict)
        for optimizer, optimizer_state in zip(self.optimizers, state_dict.get("optimizers", [])):
            optimizer.load_state_dict(optimizer_state)

    def training_step(
        self,
        batch: Mapping[str, torch.Tensor],
        scaler: torch.amp.GradScaler,
        autocast_context: Any,
        gradient_clip_norm: float | None,
    ) -> dict[str, float]:
        noise_level = self.current_noise_level()
        target = batch["target"]

        potential_optimizer = self.optimizers[1]
        potential_optimizer.zero_grad(set_to_none=True)
        with frozen_parameters(self.transport):
            with autocast_context():
                noised_source = self._perturb_source(batch["source"])
                transported = self.transport(noised_source).detach()
                potential_real = self._dual_potential(target)
                potential_fake = self._dual_potential(transported)
                potential_objective = potential_real.mean() - potential_fake.mean()
                potential_loss = -potential_objective
        scaler.scale(potential_loss).backward()
        if self.gradient_clip_enabled and gradient_clip_norm is not None:
            scaler.unscale_(potential_optimizer)
            torch.nn.utils.clip_grad_norm_(self.potential_backbone.parameters(), gradient_clip_norm)
        scaler.step(potential_optimizer)

        map_optimizer = self.optimizers[0]
        map_losses: list[float] = []
        cost_values: list[float] = []
        for _ in range(self.transport_steps):
            map_optimizer.zero_grad(set_to_none=True)
            with frozen_parameters(self.potential_backbone):
                with autocast_context():
                    noised_source = self._perturb_source(batch["source"])
                    transported = self.transport(noised_source)
                    potential_fake = self._dual_potential(transported)
                    cost = self._transport_cost(noised_source, transported).mean()
                    map_loss = cost - potential_fake.mean()
            scaler.scale(map_loss).backward()
            if self.gradient_clip_enabled and gradient_clip_norm is not None:
                scaler.unscale_(map_optimizer)
                torch.nn.utils.clip_grad_norm_(self.transport.parameters(), gradient_clip_norm)
            scaler.step(map_optimizer)
            map_losses.append(float(map_loss.detach()))
            cost_values.append(float(cost.detach()))
        scaler.update()
        self.train_step_index += 1
        predicted = self.compute_map(batch["source"])
        target_map = batch.get("ground_truth_map", batch["target"])
        return {
            "train/map_loss": sum(map_losses) / max(len(map_losses), 1),
            "train/cost": sum(cost_values) / max(len(cost_values), 1),
            "train/map_l2": float((predicted.detach() - target_map.detach()).pow(2).mean().sqrt()),
            "train/potential_objective": float(potential_objective.detach()),
            "train/noise_level": noise_level,
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
