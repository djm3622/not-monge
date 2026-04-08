"""Makkuva et al. minimax OT solver and ablations."""

from __future__ import annotations

from typing import Any, Mapping

import torch
from torch import nn

from src.evaluation.ot_metrics import (
    empirical_w2_distance,
    l2_unexplained_variance_percentage,
    maximum_mean_discrepancy,
    transport_cosine_similarity,
)
from src.models.potential import build_potential
from src.solvers.base import BaseOTSolver
from src.solvers.minimax_ot import frozen_parameters
from src.solvers.registry import register_solver


def _potential_gradient(module: nn.Module, inputs: torch.Tensor, create_graph: bool) -> torch.Tensor:
    with torch.enable_grad():
        if hasattr(module, "gradient"):
            return module.gradient(inputs, create_graph=create_graph)  # type: ignore[return-value]
        differentiable_inputs = inputs.requires_grad_(True)
        values = module(differentiable_inputs)
        return torch.autograd.grad(values.sum(), differentiable_inputs, create_graph=create_graph)[0]


def _module_penalty(module: nn.Module) -> torch.Tensor:
    if hasattr(module, "negative_weight_penalty"):
        return module.negative_weight_penalty()  # type: ignore[return-value]
    parameter = next(module.parameters(), None)
    if parameter is None:
        return torch.tensor(0.0)
    return parameter.new_tensor(0.0)


def _convexify_if_available(module: nn.Module) -> None:
    if hasattr(module, "convexify"):
        module.convexify()  # type: ignore[misc]


class MakkuvaOTSolver(BaseOTSolver):
    """Alternating minimax solver learning a map as the gradient of a scalar potential."""

    solver_group = "learned_w2"
    supports_training = True
    supports_potential = True

    def __init__(
        self,
        map_config: Mapping[str, Any],
        solver_config: Mapping[str, Any],
        training_config: Mapping[str, Any],
        *,
        solver_name: str,
    ) -> None:
        super().__init__(training_config=training_config)
        input_dim = int(map_config["input_dim"])
        output_dim = int(map_config.get("output_dim", input_dim))
        if input_dim != output_dim:
            raise ValueError("MakkuvaOTSolver requires matching input and output dimensions")

        f_cfg = dict(solver_config.get("f_potential", {}))
        g_cfg = dict(solver_config.get("g_potential", {}))
        f_cfg["input_dim"] = input_dim
        g_cfg["input_dim"] = input_dim

        self.f_potential = build_potential(f_cfg)
        self.g_potential = build_potential(g_cfg)
        self.solver_name = solver_name
        self.learning_rate = float(solver_config.get("lr", 1.0e-4))
        self.beta1 = float(solver_config.get("beta1", 0.5))
        self.beta2 = float(solver_config.get("beta2", 0.9))
        self.weight_decay = float(solver_config.get("weight_decay", 0.0))
        self.inner_steps = int(solver_config.get("inner_steps", 10))
        self.lambda_cvx = float(solver_config.get("lambda_cvx", 0.0))
        self.transport_l2_penalty = float(solver_config.get("transport_l2_penalty", 0.0))
        self.potential_l2_penalty = float(solver_config.get("potential_l2_penalty", 0.0))
        self.f_project_convex = bool(solver_config.get("f_project_convex", False))
        self.g_project_convex = bool(solver_config.get("g_project_convex", False))

    def compute_map(self, x: torch.Tensor) -> torch.Tensor:
        return _potential_gradient(self.g_potential, x, create_graph=torch.is_grad_enabled())

    def compute_potential(self, x: torch.Tensor) -> torch.Tensor | None:
        return self.g_potential(x)  # type: ignore[operator]

    def configure_optimizers(self, total_steps: int) -> None:
        del total_steps
        self.optimizers = [
            torch.optim.Adam(
                self.f_potential.parameters(),
                lr=self.learning_rate,
                betas=(self.beta1, self.beta2),
                weight_decay=self.weight_decay,
            ),
            torch.optim.Adam(
                self.g_potential.parameters(),
                lr=self.learning_rate,
                betas=(self.beta1, self.beta2),
                weight_decay=self.weight_decay,
            ),
        ]
        self.schedulers = []

    def state_dict(self, *args: Any, **kwargs: Any) -> dict[str, Any]:  # type: ignore[override]
        return {
            "f_potential": self.f_potential.state_dict(*args, **kwargs),
            "g_potential": self.g_potential.state_dict(*args, **kwargs),
            "optimizers": [optimizer.state_dict() for optimizer in self.optimizers],
        }

    def load_state_dict(self, state_dict: Mapping[str, Any], strict: bool = True) -> None:  # type: ignore[override]
        self.f_potential.load_state_dict(state_dict["f_potential"], strict=strict)
        self.g_potential.load_state_dict(state_dict["g_potential"], strict=strict)
        for optimizer, optimizer_state in zip(self.optimizers, state_dict.get("optimizers", [])):
            optimizer.load_state_dict(optimizer_state)

    def _w2_estimate(
        self,
        source: torch.Tensor,
        target: torch.Tensor,
        transported: torch.Tensor,
    ) -> torch.Tensor:
        return (
            self.f_potential(transported).view(-1)
            - self.f_potential(target).view(-1)
            - (source * transported).sum(dim=-1)
            + 0.5 * source.pow(2).sum(dim=-1)
            + 0.5 * target.pow(2).sum(dim=-1)
        ).mean()

    def training_step(
        self,
        batch: Mapping[str, torch.Tensor],
        scaler: torch.amp.GradScaler,
        autocast_context: Any,
        gradient_clip_norm: float | None,
    ) -> Mapping[str, float]:
        source = batch["source"].detach()
        target = batch["target"].detach()

        f_optimizer, g_optimizer = self.optimizers
        latest_g_loss = source.new_tensor(0.0)
        latest_penalty = source.new_tensor(0.0)
        latest_transport_penalty = source.new_tensor(0.0)

        for _ in range(self.inner_steps):
            g_optimizer.zero_grad(set_to_none=True)
            with autocast_context():
                transported = _potential_gradient(self.g_potential, source, create_graph=True)
                with frozen_parameters(self.f_potential):
                    f_transported = self.f_potential(transported).view(-1)
                g_objective = (f_transported - (source * transported).sum(dim=-1)).mean()
                transport_penalty = (
                    0.5 * self.transport_l2_penalty * transported.pow(2).sum(dim=-1).mean()
                )
                penalty = self.lambda_cvx * _module_penalty(self.g_potential)
                total_g_loss = g_objective + penalty + transport_penalty
            scaler.scale(total_g_loss).backward()
            if gradient_clip_norm is not None:
                scaler.unscale_(g_optimizer)
                torch.nn.utils.clip_grad_norm_(self.g_potential.parameters(), gradient_clip_norm)
            scaler.step(g_optimizer)
            scaler.update()
            if self.g_project_convex:
                _convexify_if_available(self.g_potential)
            latest_g_loss = g_objective.detach()
            latest_penalty = penalty.detach()
            latest_transport_penalty = transport_penalty.detach()

        f_optimizer.zero_grad(set_to_none=True)
        with autocast_context():
            transported = _potential_gradient(self.g_potential, source, create_graph=False).detach()
            potential_target = self.f_potential(target).view(-1)
            potential_transported = self.f_potential(transported).view(-1)
            potential_penalty = (
                0.5
                * self.potential_l2_penalty
                * (potential_target.pow(2).mean() + potential_transported.pow(2).mean())
            )
            f_loss = potential_target.mean() - potential_transported.mean() + potential_penalty
        scaler.scale(f_loss).backward()
        if gradient_clip_norm is not None:
            scaler.unscale_(f_optimizer)
            torch.nn.utils.clip_grad_norm_(self.f_potential.parameters(), gradient_clip_norm)
        scaler.step(f_optimizer)
        scaler.update()
        if self.f_project_convex:
            _convexify_if_available(self.f_potential)

        with torch.no_grad():
            transported = _potential_gradient(self.g_potential, source, create_graph=False).detach()
            w2_estimate = self._w2_estimate(source, target, transported).detach()
            metrics = {
                "train/f_loss": float(f_loss.detach()),
                "train/g_loss": float(latest_g_loss),
                "train/g_penalty": float(latest_penalty),
                "train/g_transport_penalty": float(latest_transport_penalty),
                "train/f_output_penalty": float(potential_penalty.detach()),
                "train/w2_estimate": float(w2_estimate),
            }
            if "ground_truth_map" in batch:
                metrics["train/map_l2"] = float(
                    (transported - batch["ground_truth_map"].detach()).pow(2).mean().sqrt().detach()
                )
        return metrics

    def validation_step(self, batch: Mapping[str, torch.Tensor]) -> Mapping[str, float]:
        source = batch["source"].detach()
        target = batch["target"].detach()
        transported = _potential_gradient(self.g_potential, source, create_graph=False).detach()

        metrics: dict[str, float] = {
            "val/pushforward_w2": empirical_w2_distance(transported, target),
            "val/mmd": maximum_mean_discrepancy(transported, target),
            "val/w2_estimate": float(self._w2_estimate(source, target, transported).detach()),
        }
        if "ground_truth_map" in batch:
            metrics["val/map_l2"] = float((transported - batch["ground_truth_map"]).pow(2).mean().sqrt().detach())
            metrics["val/l2_uvp_fwd"] = l2_unexplained_variance_percentage(
                transported,
                batch["ground_truth_map"].detach(),
                batch["target"].detach(),
            )
            metrics["val/transport_cos_fwd"] = transport_cosine_similarity(
                transported,
                batch["ground_truth_map"].detach(),
                batch["source"].detach(),
            )
        return metrics


@register_solver("makkuva_icnn_cvx")
def _build_makkuva_icnn_solver(
    model_config: Mapping[str, Any],
    solver_config: Mapping[str, Any],
    training_config: Mapping[str, Any],
) -> MakkuvaOTSolver:
    return MakkuvaOTSolver(
        model_config,
        solver_config,
        training_config,
        solver_name="makkuva_icnn_cvx",
    )


@register_solver("makkuva_mlp_ablation")
def _build_makkuva_mlp_solver(
    model_config: Mapping[str, Any],
    solver_config: Mapping[str, Any],
    training_config: Mapping[str, Any],
) -> MakkuvaOTSolver:
    return MakkuvaOTSolver(
        model_config,
        solver_config,
        training_config,
        solver_name="makkuva_mlp_ablation",
    )
