"""Cycle-consistent ICNN OT baseline."""

from __future__ import annotations

from typing import Any, Mapping

import torch
import torch.nn.functional as F

from src.models.ot_map import build_ot_map
from src.models.potential import build_potential
from src.solvers.base import BaseOTSolver
from src.solvers.minimax_ot import _potential_gradient, _sinkhorn_barycenters
from src.solvers.registry import register_solver
from src.training.losses import quadratic_cost


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


class ICNNBaselineSolver(BaseOTSolver):
    """Two-ICNN transport solver with cycle and conjugacy regularization."""

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
        input_dim = int(map_config["input_dim"])
        output_dim = int(map_config.get("output_dim", input_dim))

        forward_cfg = dict(solver_config.get("forward_potential", solver_config.get("potential", {})))
        forward_cfg["input_dim"] = input_dim
        forward_cfg.setdefault("kind", "denseicnn")
        forward_cfg.setdefault("activation", "celu")
        forward_cfg.setdefault("identity_quadratic", 1.0)
        forward_cfg.setdefault("strong_convexity", 1.0e-4)

        inverse_cfg = dict(solver_config.get("inverse_potential", solver_config.get("forward_potential", solver_config.get("potential", {}))))
        inverse_cfg["input_dim"] = output_dim
        inverse_cfg.setdefault("kind", forward_cfg.get("kind", "denseicnn"))
        inverse_cfg.setdefault("activation", forward_cfg.get("activation", "celu"))
        inverse_cfg.setdefault("identity_quadratic", forward_cfg.get("identity_quadratic", 1.0))
        inverse_cfg.setdefault("strong_convexity", forward_cfg.get("strong_convexity", 1.0e-4))

        self.forward_potential = build_potential(forward_cfg)
        self.inverse_potential = build_potential(inverse_cfg)
        self.learning_rate = float(solver_config.get("lr", solver_config.get("map_lr", training_config["optimizer"]["lr"])))
        self.plan_weight = float(solver_config.get("plan_weight", 1.0))
        self.plan_reg = float(solver_config.get("plan_reg", 0.1))
        self.transport_weight = float(solver_config.get("transport_weight", 0.05))
        self.cycle_weight = float(solver_config.get("cycle_weight", 5.0))
        self.conjugacy_weight = float(solver_config.get("conjugacy_weight", 1.0))
        self.mmd_weight = float(solver_config.get("mmd_weight", 0.5))
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
            torch.nn.init.zeros_(self.forward_correction.output.weight)
            torch.nn.init.zeros_(self.forward_correction.output.bias)
            torch.nn.init.zeros_(self.inverse_correction.output.weight)
            torch.nn.init.zeros_(self.inverse_correction.output.bias)
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
        return self.forward_potential(x)  # type: ignore[operator]

    def configure_optimizers(self, total_steps: int) -> None:
        self._configure_multi_optimizer(
            parameter_groups=[(
                list(self.forward_potential.parameters())
                + list(self.inverse_potential.parameters())
                + (list(self.forward_correction.parameters()) if self.forward_correction is not None else [])
                + (list(self.inverse_correction.parameters()) if self.inverse_correction is not None else []),
                self.learning_rate,
            )],
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

    def _transport_terms(
        self,
        source: torch.Tensor,
        target: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        forward = self.compute_map(source)
        backward = self.compute_inverse_map(target)
        cycle_source = self.compute_inverse_map(forward)
        cycle_target = self.compute_map(backward)
        conjugacy_forward = (
            self.forward_potential(source).view(-1)
            + self.inverse_potential(forward).view(-1)
            - (source * forward).sum(dim=-1)
        )
        conjugacy_backward = (
            self.inverse_potential(target).view(-1)
            + self.forward_potential(backward).view(-1)
            - (backward * target).sum(dim=-1)
        )
        if self.plan_weight > 0.0:
            barycentric_target, barycentric_source = _sinkhorn_barycenters(source, target, reg=self.plan_reg)
            plan_loss = F.mse_loss(forward, barycentric_target) + F.mse_loss(backward, barycentric_source)
        else:
            plan_loss = forward.new_tensor(0.0)
        return {
            "forward": forward,
            "backward": backward,
            "transport": quadratic_cost(source, forward).mean() + quadratic_cost(backward, target).mean(),
            "cycle": F.mse_loss(cycle_source, source) + F.mse_loss(cycle_target, target),
            "plan": plan_loss,
            "conjugacy": conjugacy_forward.pow(2).mean() + conjugacy_backward.pow(2).mean(),
            "mmd": _rbf_mmd_loss(forward, target) + _rbf_mmd_loss(backward, source),
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
            terms = self._transport_terms(batch["source"], batch["target"])
            supervision = (
                F.mse_loss(terms["forward"], batch["ground_truth_map"])
                if self.supervision_weight > 0.0 and "ground_truth_map" in batch
                else terms["forward"].new_tensor(0.0)
            )
            loss = (
                self.plan_weight * terms["plan"]
                + self.transport_weight * terms["transport"]
                + self.cycle_weight * terms["cycle"]
                + self.conjugacy_weight * terms["conjugacy"]
                + self.mmd_weight * terms["mmd"]
                + self.supervision_weight * supervision
            )
        scaler.scale(loss).backward()
        if gradient_clip_norm is not None:
            scaler.unscale_(optimizer)
            parameters = list(self.forward_potential.parameters()) + list(self.inverse_potential.parameters())
            if self.forward_correction is not None:
                parameters.extend(self.forward_correction.parameters())
            if self.inverse_correction is not None:
                parameters.extend(self.inverse_correction.parameters())
            torch.nn.utils.clip_grad_norm_(parameters, gradient_clip_norm)
        scaler.step(optimizer)
        scaler.update()
        self.schedulers[0].step()
        return {
            "train/icnn_loss": float(loss.detach()),
            "train/plan_loss": float(terms["plan"].detach()),
            "train/transport_loss": float(terms["transport"].detach()),
            "train/cycle_loss": float(terms["cycle"].detach()),
            "train/conjugacy_loss": float(terms["conjugacy"].detach()),
            "train/mmd_loss": float(terms["mmd"].detach()),
            "train/supervision_loss": float(supervision.detach()),
            "train/map_l2": float(
                (terms["forward"].detach() - batch["ground_truth_map"].detach()).pow(2).mean().sqrt().detach()
            ),
        }

    def validation_step(self, batch: Mapping[str, torch.Tensor]) -> Mapping[str, float]:
        metrics = dict(super().validation_step(batch))
        terms = self._transport_terms(batch["source"], batch["target"])
        supervision = (
            F.mse_loss(terms["forward"], batch["ground_truth_map"])
            if self.supervision_weight > 0.0 and "ground_truth_map" in batch
            else terms["forward"].new_tensor(0.0)
        )
        total = (
            self.plan_weight * terms["plan"]
            + self.transport_weight * terms["transport"]
            + self.cycle_weight * terms["cycle"]
            + self.conjugacy_weight * terms["conjugacy"]
            + self.mmd_weight * terms["mmd"]
            + self.supervision_weight * supervision
        )
        metrics["val/icnn_loss"] = float(total.detach())
        metrics["val/plan_loss"] = float(terms["plan"].detach())
        metrics["val/transport_loss"] = float(terms["transport"].detach())
        metrics["val/cycle_loss"] = float(terms["cycle"].detach())
        metrics["val/conjugacy_loss"] = float(terms["conjugacy"].detach())
        metrics["val/mmd_loss"] = float(terms["mmd"].detach())
        metrics["val/supervision_loss"] = float(supervision.detach())
        return metrics


@register_solver("icnn")
def _build_icnn_solver(
    model_config: Mapping[str, Any],
    solver_config: Mapping[str, Any],
    training_config: Mapping[str, Any],
) -> ICNNBaselineSolver:
    return ICNNBaselineSolver(model_config, solver_config, training_config)
