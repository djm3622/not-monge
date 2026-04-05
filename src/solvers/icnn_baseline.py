"""ICNN-induced transport baseline."""

from __future__ import annotations

from typing import Any, Mapping

import torch
from torch import nn

from src.models.potential import InputConvexNeuralNetwork, build_potential
from src.solvers.base import BaseOTSolver
from src.solvers.minimax_ot import frozen_parameters
from src.solvers.registry import register_solver
from src.training.losses import minimax_transport_objective, quadratic_cost


class ICNNTransportMap(nn.Module):
    """Transport map given by the gradient of a convex potential."""

    def __init__(self, potential: InputConvexNeuralNetwork) -> None:
        super().__init__()
        self.potential = potential

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        with torch.enable_grad():
            return self.potential.gradient(x, create_graph=torch.is_grad_enabled())


class ICNNBaselineSolver(BaseOTSolver):
    """Makkuva-style convex transport map with an adversarial potential."""

    solver_name = "icnn"
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
        map_potential_cfg = dict(solver_config["potential"])
        map_potential_cfg["input_dim"] = int(map_config["input_dim"])
        map_potential_cfg["kind"] = "icnn"
        self.map_potential = build_potential(map_potential_cfg)
        if not isinstance(self.map_potential, InputConvexNeuralNetwork):
            raise TypeError("ICNNBaselineSolver requires an ICNN transport potential")
        self.transport = ICNNTransportMap(self.map_potential)
        adversary_cfg = {
            "input_dim": int(map_config.get("output_dim", map_config["input_dim"])),
            "hidden_dims": list(solver_config["potential"]["hidden_dims"]),
            "activation": "silu",
            "kind": "mlp",
        }
        self.potential = build_potential(adversary_cfg)
        self.critic_steps = int(solver_config.get("critic_steps", 1))
        self.map_lr = float(solver_config.get("map_lr", training_config["optimizer"]["lr"]))
        self.potential_lr = float(
            solver_config.get("potential_lr", training_config["optimizer"]["lr"])
        )

    def compute_map(self, x: torch.Tensor) -> torch.Tensor:
        return self.transport(x)

    def compute_potential(self, x: torch.Tensor) -> torch.Tensor | None:
        return self.map_potential(x)

    def configure_optimizers(self, total_steps: int) -> None:
        self._configure_multi_optimizer(
            parameter_groups=[
                (self.map_potential.parameters(), self.map_lr),
                (self.potential.parameters(), self.potential_lr),
            ],
            total_steps=total_steps,
        )

    def state_dict(self, *args: Any, **kwargs: Any) -> dict[str, Any]:  # type: ignore[override]
        return {
            "map_potential": self.map_potential.state_dict(*args, **kwargs),
            "potential": self.potential.state_dict(*args, **kwargs),
            "optimizers": [optimizer.state_dict() for optimizer in self.optimizers],
            "schedulers": [scheduler.state_dict() for scheduler in self.schedulers],
        }

    def load_state_dict(self, state_dict: Mapping[str, Any], strict: bool = True) -> None:  # type: ignore[override]
        self.map_potential.load_state_dict(state_dict["map_potential"], strict=strict)
        self.potential.load_state_dict(state_dict["potential"], strict=strict)
        self.transport.potential = self.map_potential
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
        optimizer = self.optimizers[1]
        optimizer.zero_grad(set_to_none=True)
        with autocast_context():
            objective = self.potential(batch["target"]).mean() - self.potential(self.compute_map(batch["source"]).detach()).mean()
            loss = -objective
        scaler.scale(loss).backward()
        if gradient_clip_norm is not None:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(self.potential.parameters(), gradient_clip_norm)
        scaler.step(optimizer)
        return float(objective.detach())

    def _transport_step(
        self,
        batch: Mapping[str, torch.Tensor],
        scaler: torch.amp.GradScaler,
        autocast_context: Any,
        gradient_clip_norm: float | None,
    ) -> dict[str, float]:
        optimizer = self.optimizers[0]
        optimizer.zero_grad(set_to_none=True)
        with frozen_parameters(self.potential):
            with autocast_context():
                transported = self.compute_map(batch["source"])
                map_loss = minimax_transport_objective(
                    batch["source"],
                    transported,
                    self.potential(transported),
                )
                cost = quadratic_cost(batch["source"], transported).mean()
        scaler.scale(map_loss).backward()
        if gradient_clip_norm is not None:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(self.map_potential.parameters(), gradient_clip_norm)
        scaler.step(optimizer)
        return {
            "train/map_loss": float(map_loss.detach()),
            "train/cost": float(cost.detach()),
            "train/map_l2": float(
                (transported.detach() - batch["ground_truth_map"].detach()).pow(2).mean().sqrt().detach()
            ),
        }

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
                self._critic_step(batch, scaler, autocast_context, gradient_clip_norm)
            )
        metrics = self._transport_step(batch, scaler, autocast_context, gradient_clip_norm)
        scaler.update()
        for scheduler in self.schedulers:
            scheduler.step()
        metrics["train/critic_objective"] = sum(critic_values) / len(critic_values)
        return metrics


@register_solver("icnn")
def _build_icnn_solver(
    model_config: Mapping[str, Any],
    solver_config: Mapping[str, Any],
    training_config: Mapping[str, Any],
) -> ICNNBaselineSolver:
    return ICNNBaselineSolver(model_config, solver_config, training_config)
