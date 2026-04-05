"""Korotin-style benchmark baselines for neural OT."""

from __future__ import annotations

from typing import Any, Mapping

import numpy as np
import ot
import torch
import torch.nn.functional as F
from torch import nn

from src.models.ot_map import build_ot_map
from src.models.potential import InputConvexNeuralNetwork, PotentialMLP, build_potential
from src.solvers.base import BaseOTSolver
from src.solvers.minimax_ot import frozen_parameters
from src.solvers.registry import register_solver
from src.training.losses import quadratic_cost


def potential_gradient(
    potential: nn.Module,
    x: torch.Tensor,
    create_graph: bool,
) -> torch.Tensor:
    """Differentiate a scalar potential with respect to its input."""
    with torch.enable_grad():
        x = x.requires_grad_(True)
        if hasattr(potential, "gradient"):
            return potential.gradient(x, create_graph=create_graph)  # type: ignore[return-value]
        values = potential(x)
        return torch.autograd.grad(values.sum(), x, create_graph=create_graph)[0]


class PotentialMapSolver(BaseOTSolver):
    """Base class for solvers whose map is derived from a scalar potential."""

    def __init__(self, training_config: Mapping[str, Any]) -> None:
        super().__init__(training_config=training_config)
        self.potential: nn.Module

    def compute_potential(self, x: torch.Tensor) -> torch.Tensor | None:
        return self.potential(x)  # type: ignore[operator]


class MMOTSolver(PotentialMapSolver):
    """Three-player maximin OT with an unconstrained potential and amortized inner minimizer."""

    solver_name = "mm"
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
        potential_cfg = dict(solver_config["potential"])
        potential_cfg["input_dim"] = int(map_config["input_dim"])
        potential_cfg["kind"] = "mlp"
        self.potential = build_potential(potential_cfg)
        inner_cfg = dict(solver_config.get("inner_map", map_config))
        inner_cfg["input_dim"] = int(map_config.get("output_dim", map_config["input_dim"]))
        inner_cfg["output_dim"] = int(map_config["input_dim"])
        self.inner_map = build_ot_map(inner_cfg)
        self.potential_lr = float(solver_config.get("potential_lr", training_config["optimizer"]["lr"]))
        self.inner_lr = float(solver_config.get("inner_lr", training_config["optimizer"]["lr"]))
        self.critic_steps = int(solver_config.get("critic_steps", 1))

    def compute_map(self, x: torch.Tensor) -> torch.Tensor:
        return x - potential_gradient(self.potential, x, create_graph=torch.is_grad_enabled())  # type: ignore[arg-type]

    def _objective(self, source: torch.Tensor, target: torch.Tensor, inner: torch.Tensor) -> torch.Tensor:
        return self.potential(source).mean() + (quadratic_cost(inner, target) - self.potential(inner).view(-1)).mean()  # type: ignore[operator]

    def configure_optimizers(self, total_steps: int) -> None:
        self._configure_multi_optimizer(
            parameter_groups=[
                (self.potential.parameters(), self.potential_lr),
                (self.inner_map.parameters(), self.inner_lr),
            ],
            total_steps=total_steps,
        )

    def state_dict(self, *args: Any, **kwargs: Any) -> dict[str, Any]:  # type: ignore[override]
        return {
            "potential": self.potential.state_dict(*args, **kwargs),
            "inner_map": self.inner_map.state_dict(*args, **kwargs),
            "optimizers": [optimizer.state_dict() for optimizer in self.optimizers],
            "schedulers": [scheduler.state_dict() for scheduler in self.schedulers],
        }

    def load_state_dict(self, state_dict: Mapping[str, Any], strict: bool = True) -> None:  # type: ignore[override]
        self.potential.load_state_dict(state_dict["potential"], strict=strict)
        self.inner_map.load_state_dict(state_dict["inner_map"], strict=strict)
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
        optimizer = self.optimizers[0]
        optimizer.zero_grad(set_to_none=True)
        with autocast_context():
            inner = self.inner_map(batch["target"]).detach()
            objective = self._objective(batch["source"], batch["target"], inner)
            loss = -objective
        scaler.scale(loss).backward()
        if gradient_clip_norm is not None:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(self.potential.parameters(), gradient_clip_norm)
        scaler.step(optimizer)
        return float(objective.detach())

    def _inner_step(
        self,
        batch: Mapping[str, torch.Tensor],
        scaler: torch.amp.GradScaler,
        autocast_context: Any,
        gradient_clip_norm: float | None,
    ) -> dict[str, float]:
        optimizer = self.optimizers[1]
        optimizer.zero_grad(set_to_none=True)
        with frozen_parameters(self.potential):
            with autocast_context():
                inner = self.inner_map(batch["target"])
                objective = self._objective(batch["source"], batch["target"], inner)
                loss = objective
                transported = self.compute_map(batch["source"])
        scaler.scale(loss).backward()
        if gradient_clip_norm is not None:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(self.inner_map.parameters(), gradient_clip_norm)
        scaler.step(optimizer)
        return {
            "train/objective": float(objective.detach()),
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
    ) -> Mapping[str, float]:
        critic_values = []
        for _ in range(self.critic_steps):
            critic_values.append(
                self._critic_step(batch, scaler, autocast_context, gradient_clip_norm)
            )
        metrics = self._inner_step(batch, scaler, autocast_context, gradient_clip_norm)
        scaler.update()
        for scheduler in self.schedulers:
            scheduler.step()
        metrics["train/critic_objective"] = sum(critic_values) / len(critic_values)
        return metrics


class MMBatchOTSolver(PotentialMapSolver):
    """Batch-wise c-transform approximation baseline."""

    solver_name = "mm_b"
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
        potential_cfg = dict(solver_config["potential"])
        potential_cfg["input_dim"] = int(map_config["input_dim"])
        potential_cfg["kind"] = "mlp"
        self.potential = build_potential(potential_cfg)
        self.learning_rate = float(solver_config.get("potential_lr", training_config["optimizer"]["lr"]))

    def compute_map(self, x: torch.Tensor) -> torch.Tensor:
        return x - potential_gradient(self.potential, x, create_graph=torch.is_grad_enabled())  # type: ignore[arg-type]

    def configure_optimizers(self, total_steps: int) -> None:
        self._configure_multi_optimizer(
            parameter_groups=[(self.potential.parameters(), self.learning_rate)],
            total_steps=total_steps,
        )

    def state_dict(self, *args: Any, **kwargs: Any) -> dict[str, Any]:  # type: ignore[override]
        return {
            "potential": self.potential.state_dict(*args, **kwargs),
            "optimizers": [optimizer.state_dict() for optimizer in self.optimizers],
            "schedulers": [scheduler.state_dict() for scheduler in self.schedulers],
        }

    def load_state_dict(self, state_dict: Mapping[str, Any], strict: bool = True) -> None:  # type: ignore[override]
        self.potential.load_state_dict(state_dict["potential"], strict=strict)
        for optimizer, optimizer_state in zip(self.optimizers, state_dict.get("optimizers", [])):
            optimizer.load_state_dict(optimizer_state)
        for scheduler, scheduler_state in zip(self.schedulers, state_dict.get("schedulers", [])):
            scheduler.load_state_dict(scheduler_state)

    def training_step(
        self,
        batch: Mapping[str, torch.Tensor],
        scaler: torch.amp.GradScaler,
        autocast_context: Any,
        gradient_clip_norm: float | None,
    ) -> Mapping[str, float]:
        optimizer = self.optimizers[0]
        optimizer.zero_grad(set_to_none=True)
        with autocast_context():
            source_values = self.potential(batch["source"]).view(-1)  # type: ignore[operator]
            pairwise = 0.5 * torch.cdist(batch["source"], batch["target"]).pow(2) - source_values.unsqueeze(1)
            objective = source_values.mean() + pairwise.amin(dim=0).mean()
            loss = -objective
            transported = self.compute_map(batch["source"])
        scaler.scale(loss).backward()
        if gradient_clip_norm is not None:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(self.potential.parameters(), gradient_clip_norm)
        scaler.step(optimizer)
        scaler.update()
        self.schedulers[0].step()
        return {
            "train/objective": float(objective.detach()),
            "train/map_l2": float(
                (transported.detach() - batch["ground_truth_map"].detach()).pow(2).mean().sqrt().detach()
            ),
        }


class QCOTSolver(PotentialMapSolver):
    """Quadratic-cost dual-regression baseline."""

    solver_name = "qc"
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
        potential_cfg = dict(solver_config["potential"])
        potential_cfg["input_dim"] = int(map_config["input_dim"])
        potential_cfg["kind"] = "mlp"
        self.potential = build_potential(potential_cfg)
        self.learning_rate = float(solver_config.get("potential_lr", training_config["optimizer"]["lr"]))

    def compute_map(self, x: torch.Tensor) -> torch.Tensor:
        return x - potential_gradient(self.potential, x, create_graph=torch.is_grad_enabled())  # type: ignore[arg-type]

    def configure_optimizers(self, total_steps: int) -> None:
        self._configure_multi_optimizer(
            parameter_groups=[(self.potential.parameters(), self.learning_rate)],
            total_steps=total_steps,
        )

    def state_dict(self, *args: Any, **kwargs: Any) -> dict[str, Any]:  # type: ignore[override]
        return {
            "potential": self.potential.state_dict(*args, **kwargs),
            "optimizers": [optimizer.state_dict() for optimizer in self.optimizers],
            "schedulers": [scheduler.state_dict() for scheduler in self.schedulers],
        }

    def load_state_dict(self, state_dict: Mapping[str, Any], strict: bool = True) -> None:  # type: ignore[override]
        self.potential.load_state_dict(state_dict["potential"], strict=strict)
        for optimizer, optimizer_state in zip(self.optimizers, state_dict.get("optimizers", [])):
            optimizer.load_state_dict(optimizer_state)
        for scheduler, scheduler_state in zip(self.schedulers, state_dict.get("schedulers", [])):
            scheduler.load_state_dict(scheduler_state)

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
        a = np.full(source.shape[0], 1.0 / source.shape[0], dtype=np.float64)
        b = np.full(target.shape[0], 1.0 / target.shape[0], dtype=np.float64)
        cost_matrix = quadratic_cost(source.unsqueeze(1), target.unsqueeze(0)).detach().cpu().numpy()
        _, log = ot.emd(a, b, cost_matrix, log=True)
        dual_u = torch.from_numpy(np.asarray(log["u"], dtype=np.float32)).to(source.device)
        with autocast_context():
            prediction = self.potential(source).view(-1)  # type: ignore[operator]
            loss = F.mse_loss(prediction, dual_u)
            transported = self.compute_map(source)
        scaler.scale(loss).backward()
        if gradient_clip_norm is not None:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(self.potential.parameters(), gradient_clip_norm)
        scaler.step(optimizer)
        scaler.update()
        self.schedulers[0].step()
        return {
            "train/qc_loss": float(loss.detach()),
            "train/map_l2": float(
                (transported.detach() - batch["ground_truth_map"].detach()).pow(2).mean().sqrt().detach()
            ),
        }


class MMV2OTSolver(BaseOTSolver):
    """Two-ICNN maximin baseline with learned inverse map."""

    solver_name = "mmv2"
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
        forward_cfg = dict(solver_config["forward_potential"])
        forward_cfg["input_dim"] = int(map_config["input_dim"])
        forward_cfg["kind"] = "icnn"
        inverse_cfg = dict(solver_config.get("inverse_potential", solver_config["forward_potential"]))
        inverse_cfg["input_dim"] = int(map_config.get("output_dim", map_config["input_dim"]))
        inverse_cfg["kind"] = "icnn"
        self.forward_potential = build_potential(forward_cfg)
        self.inverse_potential = build_potential(inverse_cfg)
        self.forward_lr = float(solver_config.get("forward_lr", training_config["optimizer"]["lr"]))
        self.inverse_lr = float(solver_config.get("inverse_lr", training_config["optimizer"]["lr"]))
        self.critic_steps = int(solver_config.get("critic_steps", 1))

    def compute_map(self, x: torch.Tensor) -> torch.Tensor:
        return potential_gradient(self.forward_potential, x, create_graph=torch.is_grad_enabled())  # type: ignore[arg-type]

    def compute_inverse_map(self, y: torch.Tensor) -> torch.Tensor:
        return potential_gradient(self.inverse_potential, y, create_graph=torch.is_grad_enabled())  # type: ignore[arg-type]

    def compute_potential(self, x: torch.Tensor) -> torch.Tensor | None:
        return self.forward_potential(x)  # type: ignore[operator]

    def _residual_potential(self, x: torch.Tensor) -> torch.Tensor:
        return 0.5 * x.pow(2).sum(dim=-1, keepdim=True) - self.forward_potential(x)  # type: ignore[operator]

    def _objective(self, source: torch.Tensor, target: torch.Tensor, inverse: torch.Tensor) -> torch.Tensor:
        return self._residual_potential(source).mean() + (quadratic_cost(inverse, target) - self._residual_potential(inverse).view(-1)).mean()

    def configure_optimizers(self, total_steps: int) -> None:
        self._configure_multi_optimizer(
            parameter_groups=[
                (self.forward_potential.parameters(), self.forward_lr),
                (self.inverse_potential.parameters(), self.inverse_lr),
            ],
            total_steps=total_steps,
        )

    def state_dict(self, *args: Any, **kwargs: Any) -> dict[str, Any]:  # type: ignore[override]
        return {
            "forward_potential": self.forward_potential.state_dict(*args, **kwargs),
            "inverse_potential": self.inverse_potential.state_dict(*args, **kwargs),
            "optimizers": [optimizer.state_dict() for optimizer in self.optimizers],
            "schedulers": [scheduler.state_dict() for scheduler in self.schedulers],
        }

    def load_state_dict(self, state_dict: Mapping[str, Any], strict: bool = True) -> None:  # type: ignore[override]
        self.forward_potential.load_state_dict(state_dict["forward_potential"], strict=strict)
        self.inverse_potential.load_state_dict(state_dict["inverse_potential"], strict=strict)
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
        optimizer = self.optimizers[0]
        optimizer.zero_grad(set_to_none=True)
        with autocast_context():
            inverse = self.compute_inverse_map(batch["target"]).detach()
            objective = self._objective(batch["source"], batch["target"], inverse)
            loss = -objective
        scaler.scale(loss).backward()
        if gradient_clip_norm is not None:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(self.forward_potential.parameters(), gradient_clip_norm)
        scaler.step(optimizer)
        return float(objective.detach())

    def _inverse_step(
        self,
        batch: Mapping[str, torch.Tensor],
        scaler: torch.amp.GradScaler,
        autocast_context: Any,
        gradient_clip_norm: float | None,
    ) -> Mapping[str, float]:
        optimizer = self.optimizers[1]
        optimizer.zero_grad(set_to_none=True)
        with frozen_parameters(self.forward_potential):
            with autocast_context():
                inverse = self.compute_inverse_map(batch["target"])
                objective = self._objective(batch["source"], batch["target"], inverse)
                loss = objective
                transported = self.compute_map(batch["source"])
        scaler.scale(loss).backward()
        if gradient_clip_norm is not None:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(self.inverse_potential.parameters(), gradient_clip_norm)
        scaler.step(optimizer)
        return {
            "train/objective": float(objective.detach()),
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
    ) -> Mapping[str, float]:
        critic_values = []
        for _ in range(self.critic_steps):
            critic_values.append(self._critic_step(batch, scaler, autocast_context, gradient_clip_norm))
        metrics = self._inverse_step(batch, scaler, autocast_context, gradient_clip_norm)
        scaler.update()
        for scheduler in self.schedulers:
            scheduler.step()
        metrics["train/critic_objective"] = sum(critic_values) / len(critic_values)
        return metrics


class TW2OTSolver(BaseOTSolver):
    """Non-maximin ICNN pair baseline with cycle consistency regularization."""

    solver_name = "tw2"
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
        forward_cfg = dict(solver_config["forward_potential"])
        forward_cfg["input_dim"] = int(map_config["input_dim"])
        forward_cfg["kind"] = "icnn"
        inverse_cfg = dict(solver_config.get("inverse_potential", solver_config["forward_potential"]))
        inverse_cfg["input_dim"] = int(map_config.get("output_dim", map_config["input_dim"]))
        inverse_cfg["kind"] = "icnn"
        self.forward_potential = build_potential(forward_cfg)
        self.inverse_potential = build_potential(inverse_cfg)
        self.learning_rate = float(solver_config.get("lr", training_config["optimizer"]["lr"]))
        self.transport_weight = float(solver_config.get("transport_weight", 1.0))
        self.cycle_weight = float(solver_config.get("cycle_weight", 10.0))
        self.mmd_weight = float(solver_config.get("mmd_weight", 1.0))

    def compute_map(self, x: torch.Tensor) -> torch.Tensor:
        return potential_gradient(self.forward_potential, x, create_graph=torch.is_grad_enabled())  # type: ignore[arg-type]

    def compute_inverse_map(self, y: torch.Tensor) -> torch.Tensor:
        return potential_gradient(self.inverse_potential, y, create_graph=torch.is_grad_enabled())  # type: ignore[arg-type]

    def compute_potential(self, x: torch.Tensor) -> torch.Tensor | None:
        return self.forward_potential(x)  # type: ignore[operator]

    def configure_optimizers(self, total_steps: int) -> None:
        self._configure_multi_optimizer(
            parameter_groups=[(
                list(self.forward_potential.parameters()) + list(self.inverse_potential.parameters()),
                self.learning_rate,
            )],
            total_steps=total_steps,
        )

    def state_dict(self, *args: Any, **kwargs: Any) -> dict[str, Any]:  # type: ignore[override]
        return {
            "forward_potential": self.forward_potential.state_dict(*args, **kwargs),
            "inverse_potential": self.inverse_potential.state_dict(*args, **kwargs),
            "optimizers": [optimizer.state_dict() for optimizer in self.optimizers],
            "schedulers": [scheduler.state_dict() for scheduler in self.schedulers],
        }

    def load_state_dict(self, state_dict: Mapping[str, Any], strict: bool = True) -> None:  # type: ignore[override]
        self.forward_potential.load_state_dict(state_dict["forward_potential"], strict=strict)
        self.inverse_potential.load_state_dict(state_dict["inverse_potential"], strict=strict)
        for optimizer, optimizer_state in zip(self.optimizers, state_dict.get("optimizers", [])):
            optimizer.load_state_dict(optimizer_state)
        for scheduler, scheduler_state in zip(self.schedulers, state_dict.get("schedulers", [])):
            scheduler.load_state_dict(scheduler_state)

    def training_step(
        self,
        batch: Mapping[str, torch.Tensor],
        scaler: torch.amp.GradScaler,
        autocast_context: Any,
        gradient_clip_norm: float | None,
    ) -> Mapping[str, float]:
        optimizer = self.optimizers[0]
        optimizer.zero_grad(set_to_none=True)
        with autocast_context():
            forward = self.compute_map(batch["source"])
            backward = self.compute_inverse_map(batch["target"])
            cycle_source = self.compute_inverse_map(forward)
            cycle_target = self.compute_map(backward)
            transport = quadratic_cost(batch["source"], forward).mean() + quadratic_cost(backward, batch["target"]).mean()
            cycle = F.mse_loss(cycle_source, batch["source"]) + F.mse_loss(cycle_target, batch["target"])
            mmd = (forward.mean(dim=0) - batch["target"].mean(dim=0)).pow(2).mean() + (backward.mean(dim=0) - batch["source"].mean(dim=0)).pow(2).mean()
            loss = self.transport_weight * transport + self.cycle_weight * cycle + self.mmd_weight * mmd
        scaler.scale(loss).backward()
        if gradient_clip_norm is not None:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(
                list(self.forward_potential.parameters()) + list(self.inverse_potential.parameters()),
                gradient_clip_norm,
            )
        scaler.step(optimizer)
        scaler.update()
        self.schedulers[0].step()
        return {
            "train/tw2_loss": float(loss.detach()),
            "train/map_l2": float(
                (forward.detach() - batch["ground_truth_map"].detach()).pow(2).mean().sqrt().detach()
            ),
        }


@register_solver("mm")
def _build_mm_solver(
    model_config: Mapping[str, Any],
    solver_config: Mapping[str, Any],
    training_config: Mapping[str, Any],
) -> MMOTSolver:
    return MMOTSolver(model_config, solver_config, training_config)


@register_solver("mm_b")
def _build_mm_b_solver(
    model_config: Mapping[str, Any],
    solver_config: Mapping[str, Any],
    training_config: Mapping[str, Any],
) -> MMBatchOTSolver:
    return MMBatchOTSolver(model_config, solver_config, training_config)


@register_solver("qc")
def _build_qc_solver(
    model_config: Mapping[str, Any],
    solver_config: Mapping[str, Any],
    training_config: Mapping[str, Any],
) -> QCOTSolver:
    return QCOTSolver(model_config, solver_config, training_config)


@register_solver("mmv2")
def _build_mmv2_solver(
    model_config: Mapping[str, Any],
    solver_config: Mapping[str, Any],
    training_config: Mapping[str, Any],
) -> MMV2OTSolver:
    return MMV2OTSolver(model_config, solver_config, training_config)


@register_solver("tw2")
def _build_tw2_solver(
    model_config: Mapping[str, Any],
    solver_config: Mapping[str, Any],
    training_config: Mapping[str, Any],
) -> TW2OTSolver:
    return TW2OTSolver(model_config, solver_config, training_config)
