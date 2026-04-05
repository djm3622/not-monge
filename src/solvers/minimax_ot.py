"""Minimax neural OT solver."""

from __future__ import annotations

from contextlib import contextmanager
from typing import Any, Mapping

import torch
from torch import nn

from src.models.ot_map import build_ot_map
from src.models.potential import build_potential
from src.solvers.base import BaseOTSolver
from src.solvers.registry import register_solver
from src.training.losses import (
    minimax_potential_objective,
    minimax_transport_objective,
    quadratic_cost,
)


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
    """Alternating minimax solver for neural OT."""

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
        self.transport = build_ot_map(map_config)
        potential_cfg = dict(solver_config.get("potential", {}))
        potential_cfg["input_dim"] = int(map_config.get("output_dim", map_config["input_dim"]))
        self.potential = build_potential(potential_cfg)
        self.critic_steps = int(solver_config.get("critic_steps", 1))
        self.map_lr = float(solver_config.get("map_lr", training_config["optimizer"]["lr"]))
        self.potential_lr = float(
            solver_config.get("potential_lr", training_config["optimizer"]["lr"])
        )
        self.weight_decay = float(
            solver_config.get("weight_decay", training_config["optimizer"]["weight_decay"])
        )

    def compute_map(self, x: torch.Tensor) -> torch.Tensor:
        return self.transport(x)

    def compute_potential(self, x: torch.Tensor) -> torch.Tensor | None:
        return self.potential(x)

    def configure_optimizers(self, total_steps: int) -> None:
        self._configure_multi_optimizer(
            parameter_groups=[
                (self.transport.parameters(), self.map_lr),
                (self.potential.parameters(), self.potential_lr),
            ],
            total_steps=total_steps,
        )

    def state_dict(self, *args: Any, **kwargs: Any) -> dict[str, Any]:  # type: ignore[override]
        return {
            "transport": self.transport.state_dict(*args, **kwargs),
            "potential": self.potential.state_dict(*args, **kwargs),
            "optimizers": [optimizer.state_dict() for optimizer in self.optimizers],
            "schedulers": [scheduler.state_dict() for scheduler in self.schedulers],
        }

    def load_state_dict(self, state_dict: Mapping[str, Any], strict: bool = True) -> None:  # type: ignore[override]
        self.transport.load_state_dict(state_dict["transport"], strict=strict)
        self.potential.load_state_dict(state_dict["potential"], strict=strict)
        for optimizer, optimizer_state in zip(self.optimizers, state_dict.get("optimizers", [])):
            optimizer.load_state_dict(optimizer_state)
        for scheduler, scheduler_state in zip(self.schedulers, state_dict.get("schedulers", [])):
            scheduler.load_state_dict(scheduler_state)

    def _critic_step(
        self,
        batch: Mapping[str, torch.Tensor],
        scaler: torch.amp.GradScaler,
        autocast_context: Any,
        gradient_clip_norm: float | None,
    ) -> float:
        potential_optimizer = self.optimizers[1]
        potential_optimizer.zero_grad(set_to_none=True)
        source = batch["source"]
        target = batch["target"]
        with autocast_context():
            transported = self.compute_map(source).detach()
            critic_objective = minimax_potential_objective(
                self.potential(target),
                self.potential(transported),
            )
            loss = -critic_objective
        scaler.scale(loss).backward()
        if gradient_clip_norm is not None:
            scaler.unscale_(potential_optimizer)
            torch.nn.utils.clip_grad_norm_(self.potential.parameters(), gradient_clip_norm)
        scaler.step(potential_optimizer)
        return float(critic_objective.detach())

    def _transport_step(
        self,
        batch: Mapping[str, torch.Tensor],
        scaler: torch.amp.GradScaler,
        autocast_context: Any,
        gradient_clip_norm: float | None,
    ) -> dict[str, float]:
        map_optimizer = self.optimizers[0]
        map_optimizer.zero_grad(set_to_none=True)
        source = batch["source"]
        target = batch["target"]
        with frozen_parameters(self.potential):
            with autocast_context():
                transported = self.compute_map(source)
                potential_fake = self.potential(transported)
                map_loss = minimax_transport_objective(source, transported, potential_fake)
                cost = quadratic_cost(source, transported).mean()
        scaler.scale(map_loss).backward()
        if gradient_clip_norm is not None:
            scaler.unscale_(map_optimizer)
            torch.nn.utils.clip_grad_norm_(self.transport.parameters(), gradient_clip_norm)
        scaler.step(map_optimizer)
        metrics = {
            "train/map_loss": float(map_loss.detach()),
            "train/cost": float(cost.detach()),
            "train/map_l2": float((transported.detach() - target).pow(2).mean().sqrt()),
        }
        return metrics

    def training_step(
        self,
        batch: Mapping[str, torch.Tensor],
        scaler: torch.amp.GradScaler,
        autocast_context: Any,
        gradient_clip_norm: float | None,
    ) -> dict[str, float]:
        critic_values = []
        for _ in range(self.critic_steps):
            critic_values.append(
                self._critic_step(
                    batch=batch,
                    scaler=scaler,
                    autocast_context=autocast_context,
                    gradient_clip_norm=gradient_clip_norm,
                )
            )
        metrics = self._transport_step(
            batch=batch,
            scaler=scaler,
            autocast_context=autocast_context,
            gradient_clip_norm=gradient_clip_norm,
        )
        scaler.update()
        for scheduler in self.schedulers:
            scheduler.step()
        metrics["train/critic_objective"] = sum(critic_values) / len(critic_values)
        metrics["train/map_lr"] = self.schedulers[0].get_last_lr()[0]
        metrics["train/potential_lr"] = self.schedulers[1].get_last_lr()[0]
        return metrics

    def validation_step(self, batch: Mapping[str, torch.Tensor]) -> dict[str, float]:
        metrics = dict(super().validation_step(batch))
        transported = self.compute_map(batch["source"])
        objective = minimax_potential_objective(
            self.potential(batch["target"]),
            self.potential(transported),
        )
        metrics["val/objective"] = float(objective.detach())
        return metrics


@register_solver("minimax")
def _build_minimax_solver(
    model_config: Mapping[str, Any],
    solver_config: Mapping[str, Any],
    training_config: Mapping[str, Any],
) -> MinimaxOTSolver:
    return MinimaxOTSolver(model_config, solver_config, training_config)
