"""Minimax neural OT solver."""

from __future__ import annotations

from contextlib import contextmanager
from typing import Any, Mapping

import numpy as np
import ot
import torch
import torch.nn.functional as F
from torch import nn

from src.models.ot_map import build_ot_map
from src.models.potential import build_potential
from src.solvers.base import BaseOTSolver
from src.solvers.registry import register_solver
from src.training.losses import quadratic_cost


def _potential_gradient(module: nn.Module, x: torch.Tensor, create_graph: bool) -> torch.Tensor:
    with torch.enable_grad():
        if hasattr(module, "gradient"):
            return module.gradient(x, create_graph=create_graph)  # type: ignore[return-value]
        x = x.requires_grad_(True)
        value = module(x)
        return torch.autograd.grad(value.sum(), x, create_graph=create_graph)[0]


def _uniform_weights(num_points: int) -> tuple[np.ndarray, np.ndarray]:
    weights = np.full(num_points, 1.0 / num_points, dtype=np.float64)
    return weights, weights.copy()


def _sinkhorn_barycenters(
    source: torch.Tensor,
    target: torch.Tensor,
    reg: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    source_weights, target_weights = _uniform_weights(source.shape[0])
    cost_matrix = torch.cdist(source.detach(), target.detach()).pow(2).cpu().numpy()
    plan = ot.sinkhorn(
        source_weights,
        target_weights,
        cost_matrix,
        reg=reg,
        method="sinkhorn_log",
        numItermax=5000,
    )
    plan_tensor = torch.from_numpy(np.asarray(plan, dtype=np.float32)).to(source.device)
    row_sums = plan_tensor.sum(dim=1, keepdim=True).clamp_min(1.0e-8)
    column_sums = plan_tensor.sum(dim=0, keepdim=True).clamp_min(1.0e-8)
    barycentric_target = plan_tensor @ target / row_sums
    barycentric_source = plan_tensor.transpose(0, 1) @ source / column_sums.transpose(0, 1)
    return barycentric_target, barycentric_source


@contextmanager
def frozen_parameters(module: nn.Module):
    """Temporarily freeze a module while still allowing input gradients."""
    original = [parameter.requires_grad for parameter in module.parameters()]
    for parameter in module.parameters():
        parameter.requires_grad_(False)
    try:
        yield
    finally:
        for parameter, requires_grad in zip(module.parameters(), original):
            parameter.requires_grad_(requires_grad)


class MinimaxOTSolver(BaseOTSolver):
    """Alternating maximin solver with gradient-parameterized potentials."""

    solver_name = "minimax"
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

        forward_cfg = dict(solver_config.get("potential", {}))
        forward_cfg["input_dim"] = input_dim
        forward_cfg.setdefault("kind", "denseicnn_u")
        forward_cfg.setdefault("activation", "celu")
        forward_cfg.setdefault("identity_quadratic", 1.0)
        forward_cfg.setdefault("strong_convexity", 1.0e-4)

        inverse_cfg = dict(solver_config.get("inverse_potential", solver_config.get("potential", {})))
        inverse_cfg["input_dim"] = output_dim
        inverse_cfg.setdefault("kind", forward_cfg.get("kind", "denseicnn_u"))
        inverse_cfg.setdefault("activation", forward_cfg.get("activation", "celu"))
        inverse_cfg.setdefault("identity_quadratic", forward_cfg.get("identity_quadratic", 1.0))
        inverse_cfg.setdefault("strong_convexity", forward_cfg.get("strong_convexity", 1.0e-4))

        self.forward_potential = build_potential(forward_cfg)
        self.inverse_potential = build_potential(inverse_cfg)
        self.forward_lr = float(solver_config.get("potential_lr", training_config["optimizer"]["lr"]))
        self.inverse_lr = float(solver_config.get("inverse_lr", self.forward_lr))
        self.forward_steps = int(solver_config.get("forward_steps", 1))
        self.inverse_steps = int(solver_config.get("inverse_steps", solver_config.get("critic_steps", 1)))
        self.objective_weight = float(solver_config.get("objective_weight", 1.0))
        self.cycle_weight = float(solver_config.get("cycle_weight", 0.0))
        self.plan_weight = float(solver_config.get("plan_weight", 0.0))
        self.plan_reg = float(solver_config.get("plan_reg", 0.1))
        self.supervision_weight = float(solver_config.get("supervision_weight", 0.0))
        self.correction_enabled = bool(solver_config.get("correction_enabled", False))
        self.correction_scale = float(solver_config.get("correction_scale", 1.0))
        if self.correction_enabled:
            forward_map_cfg = dict(map_config)
            forward_map_cfg["input_dim"] = input_dim
            forward_map_cfg["output_dim"] = output_dim
            inverse_map_cfg = dict(map_config)
            inverse_map_cfg["input_dim"] = output_dim
            inverse_map_cfg["output_dim"] = input_dim
            self.forward_correction = build_ot_map(forward_map_cfg)
            self.inverse_correction = build_ot_map(inverse_map_cfg)
            nn.init.zeros_(self.forward_correction.output.weight)
            nn.init.zeros_(self.forward_correction.output.bias)
            nn.init.zeros_(self.inverse_correction.output.weight)
            nn.init.zeros_(self.inverse_correction.output.bias)
        else:
            self.forward_correction = None
            self.inverse_correction = None

    def compute_map(self, x: torch.Tensor) -> torch.Tensor:
        gradient = _potential_gradient(self.forward_potential, x, create_graph=torch.is_grad_enabled())
        if self.forward_correction is None:
            return gradient
        return gradient + self.correction_scale * self.forward_correction(x)

    def compute_inverse_map(self, y: torch.Tensor) -> torch.Tensor:
        gradient = _potential_gradient(self.inverse_potential, y, create_graph=torch.is_grad_enabled())
        if self.inverse_correction is None:
            return gradient
        return gradient + self.correction_scale * self.inverse_correction(y)

    def compute_potential(self, x: torch.Tensor) -> torch.Tensor | None:
        return self._residual_potential(x)

    def configure_optimizers(self, total_steps: int) -> None:
        self._configure_multi_optimizer(
            parameter_groups=[
                (
                    list(self.forward_potential.parameters())
                    + (list(self.forward_correction.parameters()) if self.forward_correction is not None else []),
                    self.forward_lr,
                ),
                (
                    list(self.inverse_potential.parameters())
                    + (list(self.inverse_correction.parameters()) if self.inverse_correction is not None else []),
                    self.inverse_lr,
                ),
            ],
            total_steps=total_steps,
        )

    def state_dict(self, *args: Any, **kwargs: Any) -> dict[str, Any]:  # type: ignore[override]
        return {
            "forward_potential": self.forward_potential.state_dict(*args, **kwargs),
            "inverse_potential": self.inverse_potential.state_dict(*args, **kwargs),
            "forward_correction": self.forward_correction.state_dict(*args, **kwargs) if self.forward_correction is not None else None,
            "inverse_correction": self.inverse_correction.state_dict(*args, **kwargs) if self.inverse_correction is not None else None,
            "optimizers": [optimizer.state_dict() for optimizer in self.optimizers],
            "schedulers": [scheduler.state_dict() for scheduler in self.schedulers],
        }

    def load_state_dict(self, state_dict: Mapping[str, Any], strict: bool = True) -> None:  # type: ignore[override]
        self.forward_potential.load_state_dict(state_dict["forward_potential"], strict=strict)
        self.inverse_potential.load_state_dict(state_dict["inverse_potential"], strict=strict)
        if self.forward_correction is not None and state_dict.get("forward_correction") is not None:
            self.forward_correction.load_state_dict(state_dict["forward_correction"], strict=strict)
        if self.inverse_correction is not None and state_dict.get("inverse_correction") is not None:
            self.inverse_correction.load_state_dict(state_dict["inverse_correction"], strict=strict)
        for optimizer, optimizer_state in zip(self.optimizers, state_dict.get("optimizers", [])):
            optimizer.load_state_dict(optimizer_state)
        for scheduler, scheduler_state in zip(self.schedulers, state_dict.get("schedulers", [])):
            scheduler.load_state_dict(scheduler_state)

    def _residual_potential(self, x: torch.Tensor) -> torch.Tensor:
        return 0.5 * x.pow(2).sum(dim=-1, keepdim=True) - self.forward_potential(x)  # type: ignore[operator]

    def _objective(self, source: torch.Tensor, target: torch.Tensor, inverse: torch.Tensor) -> torch.Tensor:
        return self._residual_potential(source).mean() + (
            quadratic_cost(inverse, target) - self._residual_potential(inverse).view(-1)
        ).mean()

    def _cycle_penalty(
        self,
        source: torch.Tensor,
        target: torch.Tensor,
        transported: torch.Tensor | None = None,
        inverse: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if self.cycle_weight <= 0.0:
            return source.new_tensor(0.0)
        transported = self.compute_map(source) if transported is None else transported
        inverse = self.compute_inverse_map(target) if inverse is None else inverse
        return F.mse_loss(self.compute_inverse_map(transported), source) + F.mse_loss(
            self.compute_map(inverse),
            target,
        )

    def _plan_penalty(
        self,
        transported: torch.Tensor,
        inverse: torch.Tensor,
        barycentric_target: torch.Tensor | None,
        barycentric_source: torch.Tensor | None,
    ) -> torch.Tensor:
        if self.plan_weight <= 0.0 or barycentric_target is None or barycentric_source is None:
            return transported.new_tensor(0.0)
        return F.mse_loss(transported, barycentric_target) + F.mse_loss(inverse, barycentric_source)

    def _supervision_penalty(
        self,
        transported: torch.Tensor,
        ground_truth_map: torch.Tensor | None,
    ) -> torch.Tensor:
        if self.supervision_weight <= 0.0 or ground_truth_map is None:
            return transported.new_tensor(0.0)
        return F.mse_loss(transported, ground_truth_map)

    def _forward_step(
        self,
        batch: Mapping[str, torch.Tensor],
        scaler: torch.amp.GradScaler,
        autocast_context: Any,
        gradient_clip_norm: float | None,
        barycentric_target: torch.Tensor | None,
        barycentric_source: torch.Tensor | None,
    ) -> dict[str, float]:
        optimizer = self.optimizers[0]
        optimizer.zero_grad(set_to_none=True)
        with autocast_context():
            transported = self.compute_map(batch["source"])
            inverse = self.compute_inverse_map(batch["target"]).detach()
            objective = self._objective(batch["source"], batch["target"], inverse)
            cycle_penalty = self._cycle_penalty(
                batch["source"],
                batch["target"],
                transported=transported,
                inverse=inverse,
            )
            plan_penalty = self._plan_penalty(transported, inverse, barycentric_target, barycentric_source)
            supervision_penalty = self._supervision_penalty(
                transported,
                batch.get("ground_truth_map"),
            )
            loss = -(
                self.objective_weight * objective
                - self.cycle_weight * cycle_penalty
                - self.plan_weight * plan_penalty
                - self.supervision_weight * supervision_penalty
            )
        scaler.scale(loss).backward()
        if gradient_clip_norm is not None:
            scaler.unscale_(optimizer)
            parameters = list(self.forward_potential.parameters())
            if self.forward_correction is not None:
                parameters.extend(self.forward_correction.parameters())
            torch.nn.utils.clip_grad_norm_(parameters, gradient_clip_norm)
        scaler.step(optimizer)
        return {
            "objective": float(objective.detach()),
            "cycle": float(cycle_penalty.detach()),
            "plan": float(plan_penalty.detach()),
            "supervision": float(supervision_penalty.detach()),
        }

    def _inverse_step(
        self,
        batch: Mapping[str, torch.Tensor],
        scaler: torch.amp.GradScaler,
        autocast_context: Any,
        gradient_clip_norm: float | None,
        barycentric_target: torch.Tensor | None,
        barycentric_source: torch.Tensor | None,
    ) -> dict[str, float]:
        optimizer = self.optimizers[1]
        optimizer.zero_grad(set_to_none=True)
        with frozen_parameters(self.forward_potential):
            with autocast_context():
                transported = self.compute_map(batch["source"]).detach()
                inverse = self.compute_inverse_map(batch["target"])
                objective = self._objective(batch["source"], batch["target"], inverse)
                cycle_penalty = self._cycle_penalty(
                    batch["source"],
                    batch["target"],
                    transported=transported,
                    inverse=inverse,
                )
                plan_penalty = self._plan_penalty(transported, inverse, barycentric_target, barycentric_source)
                loss = (
                    self.objective_weight * objective
                    + self.cycle_weight * cycle_penalty
                    + self.plan_weight * plan_penalty
                )
        scaler.scale(loss).backward()
        if gradient_clip_norm is not None:
            scaler.unscale_(optimizer)
            parameters = list(self.inverse_potential.parameters())
            if self.inverse_correction is not None:
                parameters.extend(self.inverse_correction.parameters())
            torch.nn.utils.clip_grad_norm_(parameters, gradient_clip_norm)
        scaler.step(optimizer)
        return {
            "objective": float(objective.detach()),
            "cycle": float(cycle_penalty.detach()),
            "plan": float(plan_penalty.detach()),
            "supervision": 0.0,
        }

    def training_step(
        self,
        batch: Mapping[str, torch.Tensor],
        scaler: torch.amp.GradScaler,
        autocast_context: Any,
        gradient_clip_norm: float | None,
    ) -> dict[str, float]:
        barycentric_target = None
        barycentric_source = None
        if self.plan_weight > 0.0:
            barycentric_target, barycentric_source = _sinkhorn_barycenters(
                batch["source"],
                batch["target"],
                reg=self.plan_reg,
            )

        forward_metrics = []
        for _ in range(self.forward_steps):
            forward_metrics.append(
                self._forward_step(
                    batch=batch,
                    scaler=scaler,
                    autocast_context=autocast_context,
                    gradient_clip_norm=gradient_clip_norm,
                    barycentric_target=barycentric_target,
                    barycentric_source=barycentric_source,
                )
            )

        inverse_metrics = []
        for _ in range(self.inverse_steps):
            inverse_metrics.append(
                self._inverse_step(
                    batch=batch,
                    scaler=scaler,
                    autocast_context=autocast_context,
                    gradient_clip_norm=gradient_clip_norm,
                    barycentric_target=barycentric_target,
                    barycentric_source=barycentric_source,
                )
            )

        scaler.update()
        for scheduler in self.schedulers:
            scheduler.step()

        transported = self.compute_map(batch["source"])
        return {
            "train/objective": sum(metric["objective"] for metric in forward_metrics) / max(len(forward_metrics), 1),
            "train/inner_objective": sum(metric["objective"] for metric in inverse_metrics) / max(len(inverse_metrics), 1),
            "train/cycle_penalty": sum(metric["cycle"] for metric in inverse_metrics) / max(len(inverse_metrics), 1),
            "train/plan_penalty": sum(metric["plan"] for metric in inverse_metrics) / max(len(inverse_metrics), 1),
            "train/supervision_penalty": sum(metric["supervision"] for metric in forward_metrics) / max(len(forward_metrics), 1),
            "train/map_l2": float((transported.detach() - batch["ground_truth_map"]).pow(2).mean().sqrt()),
            "train/forward_lr": self.schedulers[0].get_last_lr()[0],
            "train/inverse_lr": self.schedulers[1].get_last_lr()[0],
        }

    def validation_step(self, batch: Mapping[str, torch.Tensor]) -> dict[str, float]:
        metrics = dict(super().validation_step(batch))
        inverse = self.compute_inverse_map(batch["target"])
        objective = self._objective(batch["source"], batch["target"], inverse)
        cycle_penalty = self._cycle_penalty(batch["source"], batch["target"], inverse=inverse)
        supervision_penalty = self._supervision_penalty(
            self.compute_map(batch["source"]),
            batch.get("ground_truth_map"),
        )
        metrics["val/objective"] = float(objective.detach())
        metrics["val/cycle_penalty"] = float(cycle_penalty.detach())
        metrics["val/supervision_penalty"] = float(supervision_penalty.detach())
        return metrics


@register_solver("minimax")
def _build_minimax_solver(
    model_config: Mapping[str, Any],
    solver_config: Mapping[str, Any],
    training_config: Mapping[str, Any],
) -> MinimaxOTSolver:
    return MinimaxOTSolver(model_config, solver_config, training_config)
