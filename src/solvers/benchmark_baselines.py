"""Paper-faithful OT benchmark baselines."""

from __future__ import annotations

import math
from typing import Any, Mapping

import numpy as np
import ot
import torch
import torch.nn.functional as F
from torch import nn

from src.evaluation.ot_metrics import l2_unexplained_variance_percentage, transport_cosine_similarity
from src.models.potential import build_potential
from src.solvers.base import BaseOTSolver
from src.solvers.minimax_ot import frozen_parameters
from src.solvers.registry import register_solver


def potential_gradient(
    potential: nn.Module,
    x: torch.Tensor,
    create_graph: bool,
) -> torch.Tensor:
    """Differentiate a scalar potential with respect to its input."""
    with torch.enable_grad():
        if hasattr(potential, "gradient"):
            return potential.gradient(x, create_graph=create_graph)  # type: ignore[return-value]
        x = x.requires_grad_(True)
        values = potential(x)
        return torch.autograd.grad(values.sum(), x, create_graph=create_graph)[0]


def _convexify_if_available(module: nn.Module) -> None:
    if hasattr(module, "convexify"):
        module.convexify()  # type: ignore[misc]


def _identity_pretrain(
    module: nn.Module,
    input_dim: int,
    steps: int,
    batch_size: int,
    learning_rate: float,
    blow: float,
    tol: float,
    seed: int,
) -> None:
    if steps <= 0:
        return
    device = next(module.parameters()).device
    optimizer = torch.optim.Adam(module.parameters(), lr=learning_rate, weight_decay=1.0e-10)
    generator = torch.Generator().manual_seed(seed)
    module.train(True)
    for _ in range(steps):
        batch = blow * torch.randn(batch_size, input_dim, generator=generator, dtype=torch.float32).to(device)
        batch.requires_grad_(True)
        prediction = potential_gradient(module, batch, create_graph=True)
        loss = F.mse_loss(prediction, batch.detach())
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        _convexify_if_available(module)
        if float(loss.detach()) < tol:
            break


def _batch_map_l2(batch: Mapping[str, torch.Tensor], prediction: torch.Tensor) -> float:
    return float((prediction.detach() - batch["ground_truth_map"].detach()).pow(2).mean().sqrt().detach())


def _monitor_w2_term(source: torch.Tensor, target: torch.Tensor, inverse: torch.Tensor) -> torch.Tensor:
    return (
        -0.5 * source.pow(2).sum(dim=1).mean()
        + ((inverse * target).sum(dim=1) - 0.5 * inverse.pow(2).sum(dim=1)).mean()
    )


class PotentialMapSolver(BaseOTSolver):
    """Base class for solvers whose map is derived from scalar potentials."""

    def __init__(self, training_config: Mapping[str, Any]) -> None:
        super().__init__(training_config=training_config)
        self.forward_potential: nn.Module

    def compute_map(self, x: torch.Tensor) -> torch.Tensor:
        return potential_gradient(self.forward_potential, x, create_graph=torch.is_grad_enabled())

    def compute_potential(self, x: torch.Tensor) -> torch.Tensor | None:
        return self.forward_potential(x)  # type: ignore[operator]


class TwoPotentialSolver(PotentialMapSolver):
    """Shared paper-style setup for solvers using forward and inverse potentials."""

    forward_default_kind = "denseicnn"
    inverse_default_kind = "denseicnn"

    def __init__(
        self,
        map_config: Mapping[str, Any],
        solver_config: Mapping[str, Any],
        training_config: Mapping[str, Any],
        *,
        constrained: bool,
    ) -> None:
        super().__init__(training_config=training_config)
        input_dim = int(map_config["input_dim"])
        output_dim = int(map_config.get("output_dim", input_dim))
        potential_key = "forward_potential" if "forward_potential" in solver_config else "potential"
        inverse_key = "inverse_potential" if "inverse_potential" in solver_config else potential_key

        forward_cfg = dict(solver_config.get(potential_key, {}))
        inverse_cfg = dict(solver_config.get(inverse_key, solver_config.get(potential_key, {})))
        forward_cfg["input_dim"] = input_dim
        inverse_cfg["input_dim"] = output_dim
        forward_cfg.setdefault("kind", "denseicnn" if constrained else "denseicnn_u")
        inverse_cfg.setdefault("kind", "denseicnn" if constrained else "denseicnn_u")
        forward_cfg.setdefault("activation", "celu")
        inverse_cfg.setdefault("activation", forward_cfg["activation"])
        forward_cfg.setdefault("identity_quadratic", 0.0)
        inverse_cfg.setdefault("identity_quadratic", 0.0)
        forward_cfg.setdefault("strong_convexity", 1.0e-4)
        inverse_cfg.setdefault("strong_convexity", forward_cfg["strong_convexity"])
        self.forward_potential = build_potential(forward_cfg)
        self.inverse_potential = build_potential(inverse_cfg)
        self.project_after_step = bool(constrained)
        self.forward_lr = float(solver_config.get("forward_lr", solver_config.get("lr", 1.0e-3)))
        self.inverse_lr = float(solver_config.get("inverse_lr", solver_config.get("lr", self.forward_lr)))
        self.identity_pretrain_steps = int(solver_config.get("identity_pretrain_steps", 0))
        self.identity_pretrain_batch_size = int(solver_config.get("identity_pretrain_batch_size", 1024))
        self.identity_pretrain_lr = float(solver_config.get("identity_pretrain_lr", 1.0e-3))
        self.identity_pretrain_blow = float(solver_config.get("identity_pretrain_blow", 3.0))
        self.identity_pretrain_tol = float(solver_config.get("identity_pretrain_tol", 1.0e-3))
        self._identity_pretrained = False
        self._input_dim = input_dim

    def compute_inverse_map(self, y: torch.Tensor) -> torch.Tensor:
        return potential_gradient(self.inverse_potential, y, create_graph=torch.is_grad_enabled())

    def _maybe_identity_pretrain(self) -> None:
        if self._identity_pretrained or self.identity_pretrain_steps <= 0:
            return
        _identity_pretrain(
            self.forward_potential,
            input_dim=self._input_dim,
            steps=self.identity_pretrain_steps,
            batch_size=self.identity_pretrain_batch_size,
            learning_rate=self.identity_pretrain_lr,
            blow=self.identity_pretrain_blow,
            tol=self.identity_pretrain_tol,
            seed=int(self.training_config.get("seed", 1234)) + 17,
        )
        self.inverse_potential.load_state_dict(self.forward_potential.state_dict())
        self._identity_pretrained = True

    def configure_optimizers(self, total_steps: int) -> None:
        del total_steps
        self._maybe_identity_pretrain()
        self.optimizers = [
            torch.optim.Adam(self.forward_potential.parameters(), lr=self.forward_lr),
            torch.optim.Adam(self.inverse_potential.parameters(), lr=self.inverse_lr),
        ]
        self.schedulers = []

    def state_dict(self, *args: Any, **kwargs: Any) -> dict[str, Any]:  # type: ignore[override]
        return {
            "forward_potential": self.forward_potential.state_dict(*args, **kwargs),
            "inverse_potential": self.inverse_potential.state_dict(*args, **kwargs),
            "optimizers": [optimizer.state_dict() for optimizer in self.optimizers],
        }

    def load_state_dict(self, state_dict: Mapping[str, Any], strict: bool = True) -> None:  # type: ignore[override]
        self.forward_potential.load_state_dict(state_dict["forward_potential"], strict=strict)
        self.inverse_potential.load_state_dict(state_dict["inverse_potential"], strict=strict)
        for optimizer, optimizer_state in zip(self.optimizers, state_dict.get("optimizers", [])):
            optimizer.load_state_dict(optimizer_state)

    def validation_step(self, batch: Mapping[str, torch.Tensor]) -> Mapping[str, float]:
        metrics = dict(super().validation_step(batch))
        inverse = self.compute_inverse_map(batch["target"]).detach()
        cycle_source = self.compute_inverse_map(self.compute_map(batch["source"])).detach()
        cycle_target = self.compute_map(inverse).detach()
        inverse_map_l2 = float((inverse - batch["source"]).pow(2).mean().sqrt().detach())
        cycle_source_l2 = float((cycle_source - batch["source"]).pow(2).mean().sqrt().detach())
        cycle_target_l2 = float((cycle_target - batch["target"]).pow(2).mean().sqrt().detach())
        metrics.update(
            {
                "val/inverse_map_l2": inverse_map_l2,
                "val/l2_uvp_inv": l2_unexplained_variance_percentage(
                    inverse,
                    batch["source"].detach(),
                    batch["source"].detach(),
                ),
                "val/transport_cos_inv": transport_cosine_similarity(
                    inverse,
                    batch["source"].detach(),
                    batch["target"].detach(),
                ),
                "val/l2_uvp_total": float(metrics["val/l2_uvp_fwd"]) + l2_unexplained_variance_percentage(
                    inverse,
                    batch["source"].detach(),
                    batch["source"].detach(),
                ),
                "val/cycle_source_l2": cycle_source_l2,
                "val/cycle_target_l2": cycle_target_l2,
                "val/cycle_total_l2": cycle_source_l2 + cycle_target_l2,
            }
        )
        return metrics


class MMOTSolver(TwoPotentialSolver):
    """Korotin et al. tMMs: unconstrained three-player maximin with amortized inverse map."""

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
        super().__init__(map_config, solver_config, training_config, constrained=False)
        self.inner_steps = int(solver_config.get("inner_steps", solver_config.get("critic_steps", 15)))

    def _outer_objective(self, source: torch.Tensor, target: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        inverse = self.compute_inverse_map(target).detach()
        objective = (self.forward_potential(source) - self.forward_potential(inverse)).mean()
        return objective, inverse

    def _inner_objective(self, target: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        inverse = self.compute_inverse_map(target)
        objective = (self.forward_potential(inverse) - (inverse * target).sum(dim=1, keepdim=True)).mean()
        return objective, inverse

    def training_step(
        self,
        batch: Mapping[str, torch.Tensor],
        scaler: torch.amp.GradScaler,
        autocast_context: Any,
        gradient_clip_norm: float | None,
    ) -> Mapping[str, float]:
        outer_optimizer = self.optimizers[0]
        inner_optimizer = self.optimizers[1]
        outer_optimizer.zero_grad(set_to_none=True)
        with autocast_context():
            outer_objective, inverse_detached = self._outer_objective(batch["source"], batch["target"])
            outer_loss = outer_objective
        scaler.scale(outer_loss).backward()
        if gradient_clip_norm is not None:
            scaler.unscale_(outer_optimizer)
            torch.nn.utils.clip_grad_norm_(self.forward_potential.parameters(), gradient_clip_norm)
        scaler.step(outer_optimizer)

        inner_values = []
        latest_inverse = inverse_detached
        for _ in range(self.inner_steps):
            inner_optimizer.zero_grad(set_to_none=True)
            with frozen_parameters(self.forward_potential):
                with autocast_context():
                    inner_objective, latest_inverse = self._inner_objective(batch["target"])
            scaler.scale(inner_objective).backward()
            if gradient_clip_norm is not None:
                scaler.unscale_(inner_optimizer)
                torch.nn.utils.clip_grad_norm_(self.inverse_potential.parameters(), gradient_clip_norm)
            scaler.step(inner_optimizer)
            inner_values.append(float(inner_objective.detach()))
        scaler.update()

        transported = self.compute_map(batch["source"])
        w2_estimate = float((-outer_objective.detach() - _monitor_w2_term(batch["source"], batch["target"], latest_inverse.detach())).detach())
        return {
            "train/objective": float(outer_objective.detach()),
            "train/inner_objective": sum(inner_values) / max(len(inner_values), 1),
            "train/w2_estimate": w2_estimate,
            "train/map_l2": _batch_map_l2(batch, transported),
        }

    def validation_step(self, batch: Mapping[str, torch.Tensor]) -> Mapping[str, float]:
        metrics = super().validation_step(batch)
        inverse = self.compute_inverse_map(batch["target"]).detach()
        objective = (self.forward_potential(batch["source"]) - self.forward_potential(inverse)).mean()
        metrics["val/objective"] = float(objective.detach())
        metrics["val/w2_estimate"] = float((-objective.detach() - _monitor_w2_term(batch["source"], batch["target"], inverse)).detach())
        return metrics


class MMBatchOTSolver(TwoPotentialSolver):
    """Korotin et al. tMM-Bs: batch c-transform approximation baseline."""

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
        super().__init__(map_config, solver_config, training_config, constrained=False)

    def configure_optimizers(self, total_steps: int) -> None:
        del total_steps
        self._maybe_identity_pretrain()
        self.optimizers = [
            torch.optim.Adam(
                list(self.forward_potential.parameters()) + list(self.inverse_potential.parameters()),
                lr=self.forward_lr,
            )
        ]
        self.schedulers = []

    def state_dict(self, *args: Any, **kwargs: Any) -> dict[str, Any]:  # type: ignore[override]
        return {
            "forward_potential": self.forward_potential.state_dict(*args, **kwargs),
            "inverse_potential": self.inverse_potential.state_dict(*args, **kwargs),
            "optimizers": [optimizer.state_dict() for optimizer in self.optimizers],
        }

    def _approx_corr(
        self,
        potential: nn.Module,
        source: torch.Tensor,
        target: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        source_values = potential(source).view(-1, 1)
        with torch.no_grad():
            scores = source @ target.transpose(0, 1) - source_values
            indices = torch.argmax(scores, dim=0)
            inverse = source[indices]
        objective = (source_values.view(-1) - potential(inverse).view(-1)).mean()
        return objective, inverse

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
            objective_xy, inverse_xy = self._approx_corr(self.forward_potential, batch["source"], batch["target"])
            objective_yx, inverse_yx = self._approx_corr(self.inverse_potential, batch["target"], batch["source"])
            objective = 0.5 * (objective_xy + objective_yx)
            loss = objective
        scaler.scale(loss).backward()
        if gradient_clip_norm is not None:
            scaler.unscale_(optimizer)
            parameters = list(self.forward_potential.parameters()) + list(self.inverse_potential.parameters())
            torch.nn.utils.clip_grad_norm_(parameters, gradient_clip_norm)
        scaler.step(optimizer)
        scaler.update()

        transported = self.compute_map(batch["source"])
        w2_estimate_xy = -objective_xy.detach() - _monitor_w2_term(batch["source"], batch["target"], inverse_xy.detach())
        w2_estimate_yx = -objective_yx.detach() - _monitor_w2_term(batch["target"], batch["source"], inverse_yx.detach())
        return {
            "train/objective": float(objective.detach()),
            "train/w2_estimate": float((0.5 * (w2_estimate_xy + w2_estimate_yx)).detach()),
            "train/map_l2": _batch_map_l2(batch, transported),
        }


class QCOTSolver(TwoPotentialSolver):
    """Korotin et al. tQCs with discrete OT regression and OT regularization."""

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
        super().__init__(map_config, solver_config, training_config, constrained=False)
        self.k_constant = float(solver_config.get("k_constant", 1.0))
        self.gamma = float(solver_config.get("gamma", 0.1))
        self.regularization_weight = float(solver_config.get("regularization_weight", 4.0 * self.gamma))

    def configure_optimizers(self, total_steps: int) -> None:
        del total_steps
        self._maybe_identity_pretrain()
        self.optimizers = [
            torch.optim.Adam(
                list(self.forward_potential.parameters()) + list(self.inverse_potential.parameters()),
                lr=self.forward_lr,
            )
        ]
        self.schedulers = []

    def state_dict(self, *args: Any, **kwargs: Any) -> dict[str, Any]:  # type: ignore[override]
        return {
            "forward_potential": self.forward_potential.state_dict(*args, **kwargs),
            "inverse_potential": self.inverse_potential.state_dict(*args, **kwargs),
            "optimizers": [optimizer.state_dict() for optimizer in self.optimizers],
        }

    def _dual_regression_loss(
        self,
        potential: nn.Module,
        source: torch.Tensor,
        target: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        batch_size = source.shape[0]
        weights = np.full(batch_size, 1.0 / batch_size, dtype=np.float64)
        cost = 0.5 * self.k_constant * torch.cdist(target, source).pow(2).detach().cpu().numpy()
        plan = ot.emd(weights, weights, cost)
        plan_tensor = torch.from_numpy(np.asarray(plan, dtype=np.float32)).to(source.device)
        column_sums = plan_tensor.sum(dim=0).clamp_min(1.0e-8).unsqueeze(1)
        barycentric_target = plan_tensor.transpose(0, 1) @ target / column_sums

        _, log = ot.emd(weights, weights, cost, log=True)
        dual_source = torch.from_numpy(np.asarray(log["v"], dtype=np.float32)).to(source.device)

        source = source.requires_grad_(True)
        # For quadratic OT, the Brenier potential satisfies T(x) = grad phi(x)
        # with source-side dual u(x) = 0.5||x||^2 - phi(x).
        output_source = self.k_constant * (
            0.5 * source.pow(2).sum(dim=1) - potential(source).view(-1)
        )
        regression = F.mse_loss(
            output_source - output_source.mean(),
            dual_source - dual_source.mean(),
        )

        gradients = torch.autograd.grad(
            outputs=output_source.sum(),
            inputs=source,
            create_graph=True,
        )[0]
        transported = source - gradients / self.k_constant
        regularizer = F.mse_loss(transported, barycentric_target.detach())
        loss = regression + self.regularization_weight * regularizer
        return loss, barycentric_target.detach()

    def training_step(
        self,
        batch: Mapping[str, torch.Tensor],
        scaler: torch.amp.GradScaler,
        autocast_context: Any,
        gradient_clip_norm: float | None,
    ) -> Mapping[str, float]:
        del autocast_context
        optimizer = self.optimizers[0]
        optimizer.zero_grad(set_to_none=True)
        loss_xy, _ = self._dual_regression_loss(self.forward_potential, batch["source"], batch["target"])
        loss_yx, _ = self._dual_regression_loss(self.inverse_potential, batch["target"], batch["source"])
        loss = 0.5 * (loss_xy + loss_yx)
        scaler.scale(loss).backward()
        if gradient_clip_norm is not None:
            scaler.unscale_(optimizer)
            parameters = list(self.forward_potential.parameters()) + list(self.inverse_potential.parameters())
            torch.nn.utils.clip_grad_norm_(parameters, gradient_clip_norm)
        scaler.step(optimizer)
        scaler.update()

        transported = self.compute_map(batch["source"])
        return {
            "train/qc_loss": float(loss.detach()),
            "train/map_l2": _batch_map_l2(batch, transported),
        }


class MMV2OTSolver(MMOTSolver):
    """Korotin et al. tMMv2s: constrained ICNN version of the maximin solver."""

    solver_name = "mmv2"

    def __init__(
        self,
        map_config: Mapping[str, Any],
        solver_config: Mapping[str, Any],
        training_config: Mapping[str, Any],
    ) -> None:
        TwoPotentialSolver.__init__(self, map_config, solver_config, training_config, constrained=True)
        self.inner_steps = int(solver_config.get("inner_steps", solver_config.get("critic_steps", 15)))

    def training_step(
        self,
        batch: Mapping[str, torch.Tensor],
        scaler: torch.amp.GradScaler,
        autocast_context: Any,
        gradient_clip_norm: float | None,
    ) -> Mapping[str, float]:
        metrics = super().training_step(batch, scaler, autocast_context, gradient_clip_norm)
        _convexify_if_available(self.forward_potential)
        _convexify_if_available(self.inverse_potential)
        return metrics


class TW2OTSolver(TwoPotentialSolver):
    """Korotin et al. tW2s: constrained pair with cycle consistency instead of inner optimization."""

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
        super().__init__(map_config, solver_config, training_config, constrained=True)
        self.cycle_weight = float(solver_config.get("cycle_weight", int(map_config["input_dim"])))

    def configure_optimizers(self, total_steps: int) -> None:
        del total_steps
        self._maybe_identity_pretrain()
        self.optimizers = [
            torch.optim.Adam(
                list(self.forward_potential.parameters()) + list(self.inverse_potential.parameters()),
                lr=self.forward_lr,
            )
        ]
        self.schedulers = []

    def state_dict(self, *args: Any, **kwargs: Any) -> dict[str, Any]:  # type: ignore[override]
        return {
            "forward_potential": self.forward_potential.state_dict(*args, **kwargs),
            "inverse_potential": self.inverse_potential.state_dict(*args, **kwargs),
            "optimizers": [optimizer.state_dict() for optimizer in self.optimizers],
        }

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
            inverse_detached = self.compute_inverse_map(batch["target"]).detach()
            outer_objective = (self.forward_potential(batch["source"]) - self.forward_potential(inverse_detached)).mean()
            cycle_penalty = F.mse_loss(
                self.compute_map(self.compute_inverse_map(batch["target"])),
                batch["target"].detach(),
            ) + F.mse_loss(
                self.compute_inverse_map(self.compute_map(batch["source"])),
                batch["source"].detach(),
            )
            loss = outer_objective + self.cycle_weight * cycle_penalty
        scaler.scale(loss).backward()
        if gradient_clip_norm is not None:
            scaler.unscale_(optimizer)
            parameters = list(self.forward_potential.parameters()) + list(self.inverse_potential.parameters())
            torch.nn.utils.clip_grad_norm_(parameters, gradient_clip_norm)
        scaler.step(optimizer)
        scaler.update()
        _convexify_if_available(self.forward_potential)
        _convexify_if_available(self.inverse_potential)

        transported = self.compute_map(batch["source"])
        w2_estimate = float((-outer_objective.detach() - _monitor_w2_term(batch["source"], batch["target"], inverse_detached)).detach())
        return {
            "train/objective": float(outer_objective.detach()),
            "train/cycle_loss": float(cycle_penalty.detach()),
            "train/w2_estimate": w2_estimate,
            "train/map_l2": _batch_map_l2(batch, transported),
        }

    def validation_step(self, batch: Mapping[str, torch.Tensor]) -> Mapping[str, float]:
        metrics = super().validation_step(batch)
        inverse = self.compute_inverse_map(batch["target"]).detach()
        objective = (self.forward_potential(batch["source"]) - self.forward_potential(inverse)).mean()
        cycle_penalty = F.mse_loss(
            self.compute_map(self.compute_inverse_map(batch["target"])),
            batch["target"].detach(),
        ) + F.mse_loss(
            self.compute_inverse_map(self.compute_map(batch["source"])),
            batch["source"].detach(),
        )
        metrics["val/objective"] = float(objective.detach())
        metrics["val/cycle_loss"] = float(cycle_penalty.detach())
        metrics["val/w2_estimate"] = float((-objective.detach() - _monitor_w2_term(batch["source"], batch["target"], inverse)).detach())
        return metrics


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
