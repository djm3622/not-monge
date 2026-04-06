"""Shared abstractions for OT solver baselines."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Iterable, Mapping, Protocol

import torch
from torch import nn

from src.evaluation.ot_metrics import (
    l2_unexplained_variance_percentage,
    transport_cosine_similarity,
)
from src.training.losses import quadratic_cost
from src.training.schedulers import build_one_cycle_schedulers
from src.utils.device import maybe_compile_module


class OTSolver(Protocol):
    """Benchmark-facing OT solver interface."""

    solver_name: str
    solver_group: str
    supports_training: bool
    supports_potential: bool

    def configure_optimizers(self, total_steps: int) -> None:
        ...

    def training_step(
        self,
        batch: Mapping[str, torch.Tensor],
        scaler: torch.amp.GradScaler,
        autocast_context: Any,
        gradient_clip_norm: float | None,
    ) -> Mapping[str, float]:
        ...

    def validation_step(self, batch: Mapping[str, torch.Tensor]) -> Mapping[str, float]:
        ...

    def compute_map(self, x: torch.Tensor) -> torch.Tensor:
        ...

    def compute_potential(self, x: torch.Tensor) -> torch.Tensor | None:
        ...

    def fit_reference(
        self,
        train_loader: Any,
        val_loader: Any | None = None,
    ) -> None:
        ...


class BaseOTSolver(nn.Module, ABC):
    """Common functionality shared by learned and reference baselines."""

    solver_name: str = "base"
    solver_group: str = "learned_w2"
    supports_training: bool = True
    supports_potential: bool = True

    def __init__(self, training_config: Mapping[str, Any]) -> None:
        super().__init__()
        self.training_config = dict(training_config)
        self.compile_enabled = bool(training_config.get("compile", False))
        self.optimizers: list[torch.optim.Optimizer] = []
        self.schedulers: list[torch.optim.lr_scheduler.OneCycleLR] = []

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.compute_map(x)

    @abstractmethod
    def compute_map(self, x: torch.Tensor) -> torch.Tensor:
        """Return the learned transport map."""

    def compute_potential(self, x: torch.Tensor) -> torch.Tensor | None:
        """Return the learned scalar potential when available."""
        return None

    def fit_reference(self, train_loader: Any, val_loader: Any | None = None) -> None:
        """Fit a non-gradient reference solver."""
        return None

    def compile_modules(self) -> None:
        """Compile submodules when requested."""
        for name, child in list(self.named_children()):
            setattr(self, name, maybe_compile_module(child, self.compile_enabled))

    def configure_optimizers(self, total_steps: int) -> None:
        """Default no-op for reference solvers."""
        return None

    def validation_step(self, batch: Mapping[str, torch.Tensor]) -> Mapping[str, float]:
        """Shared validation metrics for OT recovery."""
        transported = self.compute_map(batch["source"])
        metrics = {
            "val/map_l2": float((transported - batch["ground_truth_map"]).pow(2).mean().sqrt().detach()),
            "val/pushforward_w2": float(quadratic_cost(transported, batch["target"]).mean().sqrt().detach()),
            "val/l2_uvp_fwd": l2_unexplained_variance_percentage(
                transported.detach(),
                batch["ground_truth_map"].detach(),
                batch["target"].detach(),
            ),
            "val/transport_cos_fwd": transport_cosine_similarity(
                transported.detach(),
                batch["ground_truth_map"].detach(),
                batch["source"].detach(),
            ),
        }
        potential = self.compute_potential(batch["target"])
        if potential is not None:
            metrics["val/potential_mean"] = float(potential.mean().detach())
        return metrics

    def _configure_multi_optimizer(
        self,
        parameter_groups: Iterable[tuple[Iterable[nn.Parameter], float]],
        total_steps: int,
    ) -> None:
        optimizer_cfg = self.training_config["optimizer"]
        self.optimizers = [
            torch.optim.AdamW(
                parameters,
                lr=max_lr,
                weight_decay=float(optimizer_cfg["weight_decay"]),
            )
            for parameters, max_lr in parameter_groups
        ]
        self.schedulers = build_one_cycle_schedulers(
            optimizers=self.optimizers,
            total_steps=total_steps,
            max_lrs=[max_lr for _, max_lr in parameter_groups],
            pct_start=float(optimizer_cfg["pct_start"]),
            div_factor=float(optimizer_cfg["div_factor"]),
            final_div_factor=float(optimizer_cfg["final_div_factor"]),
        )


class ReferenceOTSolver(BaseOTSolver):
    """Base class for one-shot reference solvers."""

    solver_group = "reference"
    supports_training = False

    def training_step(
        self,
        batch: Mapping[str, torch.Tensor],
        scaler: torch.amp.GradScaler,
        autocast_context: Any,
        gradient_clip_norm: float | None,
    ) -> Mapping[str, float]:
        raise RuntimeError(f"{self.solver_name} is a reference solver and does not support training_step")
