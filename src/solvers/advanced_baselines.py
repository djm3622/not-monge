"""Advanced alternative OT formulations."""

from __future__ import annotations

from typing import Any, Mapping

import numpy as np
import ot
import torch
import torch.nn.functional as F

from src.models.groupsort import GroupSortMLP
from src.models.ot_map import build_ot_map
from src.solvers.base import BaseOTSolver
from src.solvers.benchmark_baselines import potential_gradient
from src.solvers.minimax_ot import frozen_parameters
from src.solvers.registry import register_solver
from src.training.losses import quadratic_cost


class EntropicOTSolver(BaseOTSolver):
    """Entropic OT map regressed against batchwise Sinkhorn barycenters."""

    solver_name = "entropic"
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
        self.transport = build_ot_map(map_config)
        self.learning_rate = float(solver_config.get("lr", training_config["optimizer"]["lr"]))
        self.reg = float(solver_config.get("reg", 0.1))
        self.cost_weight = float(solver_config.get("cost_weight", 0.1))

    def compute_map(self, x: torch.Tensor) -> torch.Tensor:
        return self.transport(x)

    def configure_optimizers(self, total_steps: int) -> None:
        self._configure_multi_optimizer(
            parameter_groups=[(self.transport.parameters(), self.learning_rate)],
            total_steps=total_steps,
        )

    def state_dict(self, *args: Any, **kwargs: Any) -> dict[str, Any]:  # type: ignore[override]
        return {
            "transport": self.transport.state_dict(*args, **kwargs),
            "optimizers": [optimizer.state_dict() for optimizer in self.optimizers],
            "schedulers": [scheduler.state_dict() for scheduler in self.schedulers],
        }

    def load_state_dict(self, state_dict: Mapping[str, Any], strict: bool = True) -> None:  # type: ignore[override]
        self.transport.load_state_dict(state_dict["transport"], strict=strict)
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
        cost_matrix = torch.cdist(source, target).pow(2).detach().cpu().numpy()
        plan = ot.sinkhorn(a, b, cost_matrix, reg=self.reg, method="sinkhorn_log")
        plan_tensor = torch.from_numpy(np.asarray(plan, dtype=np.float32)).to(source.device)
        row_sums = plan_tensor.sum(dim=1, keepdim=True).clamp_min(1.0e-8)
        barycentric = plan_tensor @ target / row_sums
        with autocast_context():
            transported = self.compute_map(source)
            loss = F.mse_loss(transported, barycentric) + self.cost_weight * quadratic_cost(source, transported).mean()
        scaler.scale(loss).backward()
        if gradient_clip_norm is not None:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(self.transport.parameters(), gradient_clip_norm)
        scaler.step(optimizer)
        scaler.update()
        self.schedulers[0].step()
        return {
            "train/entropic_loss": float(loss.detach()),
            "train/map_l2": float(
                (transported.detach() - batch["ground_truth_map"].detach()).pow(2).mean().sqrt().detach()
            ),
        }


class W1OTSolver(BaseOTSolver):
    """Two-stage W1 OT solver with GroupSort critic and learned step size."""

    solver_name = "w1"
    solver_group = "advanced_alt"
    supports_training = True
    supports_potential = True

    def __init__(
        self,
        map_config: Mapping[str, Any],
        solver_config: Mapping[str, Any],
        training_config: Mapping[str, Any],
    ) -> None:
        super().__init__(training_config=training_config)
        hidden_dims = list(solver_config["potential"]["hidden_dims"])
        group_size = int(solver_config["potential"].get("group_size", 2))
        self.critic = GroupSortMLP(
            input_dim=int(map_config["input_dim"]),
            hidden_dims=hidden_dims,
            output_dim=1,
            group_size=group_size,
        )
        step_cfg = dict(solver_config.get("step_map", map_config))
        step_cfg["input_dim"] = int(map_config["input_dim"])
        step_cfg["output_dim"] = 1
        self.step_network = build_ot_map(step_cfg)
        self.critic_lr = float(solver_config.get("critic_lr", training_config["optimizer"]["lr"]))
        self.step_lr = float(solver_config.get("step_lr", training_config["optimizer"]["lr"]))
        self.phase1_ratio = float(solver_config.get("phase1_ratio", 0.5))
        self.gradient_penalty_weight = float(solver_config.get("gradient_penalty_weight", 10.0))
        self.train_step_index = 0
        self.phase1_steps = 1

    def compute_potential(self, x: torch.Tensor) -> torch.Tensor | None:
        return self.critic(x)

    def _critic_direction(self, x: torch.Tensor, create_graph: bool) -> torch.Tensor:
        direction = potential_gradient(self.critic, x, create_graph=create_graph)
        return F.normalize(direction, dim=-1)

    def compute_map(self, x: torch.Tensor) -> torch.Tensor:
        direction = self._critic_direction(x, create_graph=torch.is_grad_enabled())
        steps = F.softplus(self.step_network(x))
        return x + steps * direction

    def configure_optimizers(self, total_steps: int) -> None:
        self.phase1_steps = max(1, int(total_steps * self.phase1_ratio))
        phase2_steps = max(1, total_steps - self.phase1_steps)
        optimizer_cfg = self.training_config["optimizer"]
        critic_optimizer = torch.optim.AdamW(
            self.critic.parameters(),
            lr=self.critic_lr,
            weight_decay=float(optimizer_cfg["weight_decay"]),
        )
        step_optimizer = torch.optim.AdamW(
            self.step_network.parameters(),
            lr=self.step_lr,
            weight_decay=float(optimizer_cfg["weight_decay"]),
        )
        self.optimizers = [critic_optimizer, step_optimizer]
        self.schedulers = [
            torch.optim.lr_scheduler.OneCycleLR(
                critic_optimizer,
                max_lr=self.critic_lr,
                total_steps=self.phase1_steps,
                pct_start=float(optimizer_cfg["pct_start"]),
                div_factor=float(optimizer_cfg["div_factor"]),
                final_div_factor=float(optimizer_cfg["final_div_factor"]),
            ),
            torch.optim.lr_scheduler.OneCycleLR(
                step_optimizer,
                max_lr=self.step_lr,
                total_steps=phase2_steps,
                pct_start=float(optimizer_cfg["pct_start"]),
                div_factor=float(optimizer_cfg["div_factor"]),
                final_div_factor=float(optimizer_cfg["final_div_factor"]),
            ),
        ]

    def state_dict(self, *args: Any, **kwargs: Any) -> dict[str, Any]:  # type: ignore[override]
        return {
            "critic": self.critic.state_dict(*args, **kwargs),
            "step_network": self.step_network.state_dict(*args, **kwargs),
            "phase1_steps": self.phase1_steps,
            "train_step_index": self.train_step_index,
            "optimizers": [optimizer.state_dict() for optimizer in self.optimizers],
            "schedulers": [scheduler.state_dict() for scheduler in self.schedulers],
        }

    def load_state_dict(self, state_dict: Mapping[str, Any], strict: bool = True) -> None:  # type: ignore[override]
        self.critic.load_state_dict(state_dict["critic"], strict=strict)
        self.step_network.load_state_dict(state_dict["step_network"], strict=strict)
        self.phase1_steps = int(state_dict.get("phase1_steps", 1))
        self.train_step_index = int(state_dict.get("train_step_index", 0))
        for optimizer, optimizer_state in zip(self.optimizers, state_dict.get("optimizers", [])):
            optimizer.load_state_dict(optimizer_state)
        for scheduler, scheduler_state in zip(self.schedulers, state_dict.get("schedulers", [])):
            scheduler.load_state_dict(scheduler_state)

    def _gradient_penalty(self, source: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        alpha = torch.rand(source.shape[0], 1, device=source.device)
        interpolated = alpha * source + (1.0 - alpha) * target
        interpolated.requires_grad_(True)
        with torch.enable_grad():
            critic_value = self.critic(interpolated)
            gradient = torch.autograd.grad(critic_value.sum(), interpolated, create_graph=True)[0]
        return (gradient.norm(dim=-1) - 1.0).pow(2).mean()

    def training_step(
        self,
        batch: Mapping[str, torch.Tensor],
        scaler: torch.amp.GradScaler,
        autocast_context: Any,
        gradient_clip_norm: float | None,
    ) -> Mapping[str, float]:
        source = batch["source"]
        target = batch["target"]

        if self.train_step_index < self.phase1_steps:
            optimizer = self.optimizers[0]
            optimizer.zero_grad(set_to_none=True)
            with autocast_context():
                objective = self.critic(target).mean() - self.critic(source).mean()
                penalty = self.gradient_penalty_weight * self._gradient_penalty(source, target)
                loss = -objective + penalty
            scaler.scale(loss).backward()
            if gradient_clip_norm is not None:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(self.critic.parameters(), gradient_clip_norm)
            scaler.step(optimizer)
            scaler.update()
            self.schedulers[0].step()
            self.train_step_index += 1
            return {
                "train/w1_objective": float(objective.detach()),
                "train/phase": 1.0,
            }

        optimizer = self.optimizers[1]
        optimizer.zero_grad(set_to_none=True)
        a = np.full(source.shape[0], 1.0 / source.shape[0], dtype=np.float64)
        b = np.full(target.shape[0], 1.0 / target.shape[0], dtype=np.float64)
        cost_matrix = torch.cdist(source, target, p=1).detach().cpu().numpy()
        plan, _ = ot.emd(a, b, cost_matrix, log=True)
        plan_tensor = torch.from_numpy(np.asarray(plan, dtype=np.float32)).to(source.device)
        row_sums = plan_tensor.sum(dim=1, keepdim=True).clamp_min(1.0e-8)
        barycentric = plan_tensor @ target / row_sums
        with frozen_parameters(self.critic):
            with autocast_context():
                transported = self.compute_map(source)
                loss = F.mse_loss(transported, barycentric)
        scaler.scale(loss).backward()
        if gradient_clip_norm is not None:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(self.step_network.parameters(), gradient_clip_norm)
        scaler.step(optimizer)
        scaler.update()
        self.schedulers[1].step()
        self.train_step_index += 1
        return {
            "train/w1_step_loss": float(loss.detach()),
            "train/phase": 2.0,
            "train/map_l2": float(
                (transported.detach() - batch["ground_truth_map"].detach()).pow(2).mean().sqrt().detach()
            ),
        }


@register_solver("entropic")
def _build_entropic_solver(
    model_config: Mapping[str, Any],
    solver_config: Mapping[str, Any],
    training_config: Mapping[str, Any],
) -> EntropicOTSolver:
    return EntropicOTSolver(model_config, solver_config, training_config)


@register_solver("w1")
def _build_w1_solver(
    model_config: Mapping[str, Any],
    solver_config: Mapping[str, Any],
    training_config: Mapping[str, Any],
) -> W1OTSolver:
    return W1OTSolver(model_config, solver_config, training_config)
