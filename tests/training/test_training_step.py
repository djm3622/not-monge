from __future__ import annotations

import copy

import pytest
import torch

from src.solvers.registry import build_solver
from src.training.diffusion_task import DiffusionTrainingTask

pytestmark = pytest.mark.integration

LEARNED_SOLVERS = [
    "minimax",
    "icnn",
    "makkuva_icnn_cvx",
    "makkuva_mlp_ablation",
    "tw2",
    "mmv2",
    "mm",
    "mm_b",
    "qc",
    "entropic",
    "otp",
    "flow",
    "w1",
]


def _parameter_snapshot(module: torch.nn.Module) -> list[torch.Tensor]:
    return [parameter.detach().clone() for parameter in module.parameters() if parameter.requires_grad]


def _changed_parameter_count(
    module: torch.nn.Module,
    snapshot: list[torch.Tensor],
) -> int:
    current = [parameter.detach() for parameter in module.parameters() if parameter.requires_grad]
    return sum(not torch.equal(before, after) for before, after in zip(snapshot, current))


@pytest.mark.parametrize("solver_name", LEARNED_SOLVERS)
def test_learned_solver_training_step_updates_parameters(
    solver_name: str,
    solver_config_factory: object,
    ot_model_config: dict[str, object],
    tiny_training_config: dict[str, object],
    ot_batch: dict[str, torch.Tensor],
    disabled_grad_scaler: torch.amp.GradScaler,
    null_autocast: object,
) -> None:
    solver = build_solver(
        ot_model_config,
        solver_config_factory(solver_name),  # type: ignore[operator]
        tiny_training_config,
    )
    solver.configure_optimizers(total_steps=2)
    before = _parameter_snapshot(solver)
    metrics = solver.training_step(
        ot_batch,
        scaler=disabled_grad_scaler,
        autocast_context=null_autocast,
        gradient_clip_norm=1.0,
    )
    assert metrics
    assert all(torch.isfinite(torch.tensor(value)) for value in metrics.values())
    assert _changed_parameter_count(solver, before) > 0


def test_diffusion_training_step_updates_parameters(
    tiny_diffusion_model_config: dict[str, object],
    tiny_training_config: dict[str, object],
    diffusion_batch: dict[str, torch.Tensor],
    disabled_grad_scaler: torch.amp.GradScaler,
    null_autocast: object,
) -> None:
    task = DiffusionTrainingTask(
        model_config=copy.deepcopy(tiny_diffusion_model_config),
        training_config=copy.deepcopy(tiny_training_config),
    )
    task.configure_optimizers(total_steps=2)
    before = _parameter_snapshot(task.model)
    metrics = task.training_step(
        diffusion_batch,
        scaler=disabled_grad_scaler,
        autocast_context=null_autocast,
        gradient_clip_norm=1.0,
    )
    assert "train/ddpm_loss" in metrics
    assert torch.isfinite(torch.tensor(metrics["train/ddpm_loss"]))
    assert _changed_parameter_count(task.model, before) > 0
