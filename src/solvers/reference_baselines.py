"""Classical OT reference baselines."""

from __future__ import annotations

from typing import Any, Mapping

import numpy as np
import ot
import torch

from src.solvers.base import ReferenceOTSolver
from src.solvers.registry import register_solver
from src.utils.data import collect_loader_tensors
from src.utils.linalg import gaussian_ot_linear_map


class SinkhornReferenceSolver(ReferenceOTSolver):
    """Discrete Sinkhorn OT with differentiable kernel interpolation."""

    solver_name = "sinkhorn"
    supports_potential = False

    def __init__(
        self,
        model_config: Mapping[str, Any],
        solver_config: Mapping[str, Any],
        training_config: Mapping[str, Any],
    ) -> None:
        super().__init__(training_config=training_config)
        self.reg = float(solver_config.get("reg", 0.05))
        self.fit_samples = int(solver_config.get("fit_samples", 2048))
        self.kernel_bandwidth = float(solver_config.get("kernel_bandwidth", 0.0))
        self.source_support = torch.empty(0)
        self.target_support = torch.empty(0)
        self.barycentric_targets = torch.empty(0)

    def fit_reference(self, train_loader: Any, val_loader: Any | None = None) -> None:
        tensors = collect_loader_tensors(train_loader, keys=["source", "target"], max_items=self.fit_samples)
        self.source_support = tensors["source"].float()
        self.target_support = tensors["target"].float()
        a = np.full(self.source_support.shape[0], 1.0 / self.source_support.shape[0], dtype=np.float64)
        b = np.full(self.target_support.shape[0], 1.0 / self.target_support.shape[0], dtype=np.float64)
        cost_matrix = torch.cdist(self.source_support, self.target_support).pow(2).cpu().numpy()
        plan = ot.sinkhorn(a, b, cost_matrix, reg=self.reg, method="sinkhorn_log", numItermax=5000)
        plan_tensor = torch.from_numpy(np.asarray(plan, dtype=np.float32))
        row_sums = plan_tensor.sum(dim=1, keepdim=True).clamp_min(1.0e-8)
        self.barycentric_targets = plan_tensor @ self.target_support / row_sums
        if self.kernel_bandwidth <= 0.0:
            distances = torch.cdist(self.source_support, self.source_support).pow(2)
            self.kernel_bandwidth = float(distances.median().clamp_min(1.0e-4))

    def compute_map(self, x: torch.Tensor) -> torch.Tensor:
        support = self.source_support.to(x.device)
        barycentric = self.barycentric_targets.to(x.device)
        distances = torch.cdist(x, support)
        nearest = distances.argmin(dim=1)
        exact_match = distances.gather(1, nearest.unsqueeze(1)).squeeze(1) < 1.0e-8
        bandwidth = max(self.kernel_bandwidth, 1.0e-4)
        weights = torch.softmax(-distances.pow(2) / (2.0 * bandwidth), dim=1)
        mapped = weights @ barycentric
        if exact_match.any():
            mapped[exact_match] = barycentric[nearest[exact_match]]
        return mapped

    def state_dict(self, *args: Any, **kwargs: Any) -> dict[str, Any]:  # type: ignore[override]
        return {
            "source_support": self.source_support,
            "target_support": self.target_support,
            "barycentric_targets": self.barycentric_targets,
            "reg": self.reg,
            "kernel_bandwidth": self.kernel_bandwidth,
        }

    def load_state_dict(self, state_dict: Mapping[str, Any], strict: bool = True) -> None:  # type: ignore[override]
        self.source_support = state_dict["source_support"]
        self.target_support = state_dict["target_support"]
        self.barycentric_targets = state_dict["barycentric_targets"]
        self.reg = float(state_dict["reg"])
        self.kernel_bandwidth = float(state_dict["kernel_bandwidth"])


class GaussianReferenceSolver(ReferenceOTSolver):
    """Closed-form Gaussian OT baseline."""

    solver_name = "gaussian"
    supports_potential = False

    def __init__(
        self,
        model_config: Mapping[str, Any],
        solver_config: Mapping[str, Any],
        training_config: Mapping[str, Any],
    ) -> None:
        super().__init__(training_config=training_config)
        self.fit_samples = int(solver_config.get("fit_samples", 4096))
        self.source_mean = torch.empty(0)
        self.target_mean = torch.empty(0)
        self.linear_map = torch.empty(0)

    def fit_reference(self, train_loader: Any, val_loader: Any | None = None) -> None:
        tensors = collect_loader_tensors(train_loader, keys=["source", "target"], max_items=self.fit_samples)
        source = tensors["source"].float()
        target = tensors["target"].float()
        self.source_mean = source.mean(dim=0)
        self.target_mean = target.mean(dim=0)
        source_centered = source - self.source_mean
        target_centered = target - self.target_mean
        source_cov = source_centered.t() @ source_centered / max(source.shape[0] - 1, 1)
        target_cov = target_centered.t() @ target_centered / max(target.shape[0] - 1, 1)
        self.linear_map = gaussian_ot_linear_map(source_cov, target_cov)

    def compute_map(self, x: torch.Tensor) -> torch.Tensor:
        source_mean = self.source_mean.to(x.device)
        target_mean = self.target_mean.to(x.device)
        linear_map = self.linear_map.to(x.device)
        centered = x - source_mean.unsqueeze(0)
        return target_mean.unsqueeze(0) + centered @ linear_map.transpose(0, 1)

    def state_dict(self, *args: Any, **kwargs: Any) -> dict[str, Any]:  # type: ignore[override]
        return {
            "source_mean": self.source_mean,
            "target_mean": self.target_mean,
            "linear_map": self.linear_map,
        }

    def load_state_dict(self, state_dict: Mapping[str, Any], strict: bool = True) -> None:  # type: ignore[override]
        self.source_mean = state_dict["source_mean"]
        self.target_mean = state_dict["target_mean"]
        self.linear_map = state_dict["linear_map"]


@register_solver("sinkhorn")
def _build_sinkhorn_solver(
    model_config: Mapping[str, Any],
    solver_config: Mapping[str, Any],
    training_config: Mapping[str, Any],
) -> SinkhornReferenceSolver:
    return SinkhornReferenceSolver(model_config, solver_config, training_config)


@register_solver("gaussian")
def _build_gaussian_solver(
    model_config: Mapping[str, Any],
    solver_config: Mapping[str, Any],
    training_config: Mapping[str, Any],
) -> GaussianReferenceSolver:
    return GaussianReferenceSolver(model_config, solver_config, training_config)
