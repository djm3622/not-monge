from __future__ import annotations

import copy
import statistics

import pytest
import torch

from src.solvers.registry import build_solver
from src.utils.seed import seed_all

pytestmark = pytest.mark.integration


def _changed_parameter_count(
    module: torch.nn.Module,
    snapshot: list[torch.Tensor],
) -> int:
    current = [parameter.detach() for parameter in module.parameters() if parameter.requires_grad]
    return sum(not torch.equal(before, after) for before, after in zip(snapshot, current))


@pytest.mark.parametrize("solver_name", ["otp", "flow"])
def test_new_solver_single_training_step_updates_parameters(
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
    before = [parameter.detach().clone() for parameter in solver.parameters() if parameter.requires_grad]
    metrics = solver.training_step(
        ot_batch,
        scaler=disabled_grad_scaler,
        autocast_context=null_autocast,
        gradient_clip_norm=1.0,
    )
    assert metrics
    assert all(torch.isfinite(torch.tensor(value)) for value in metrics.values())
    assert _changed_parameter_count(solver, before) > 0


@pytest.mark.parametrize("solver_name", ["otp", "flow"])
def test_new_solver_short_loop_is_stable(
    solver_name: str,
    solver_config_factory: object,
    ot_model_config: dict[str, object],
    tiny_training_config: dict[str, object],
    easy_ot_batch: dict[str, torch.Tensor],
    disabled_grad_scaler: torch.amp.GradScaler,
    null_autocast: object,
) -> None:
    training_config = copy.deepcopy(tiny_training_config)
    training_config["optimizer"]["lr"] = 5.0e-3
    training_config["max_steps"] = 8
    solver_config = solver_config_factory(solver_name)  # type: ignore[operator]
    if "map_lr" in solver_config:
        solver_config["map_lr"] = 5.0e-3
    if "potential_lr" in solver_config:
        solver_config["potential_lr"] = 5.0e-3
    if "lr" in solver_config:
        solver_config["lr"] = 5.0e-3
    solver = build_solver(ot_model_config, solver_config, training_config)
    solver.configure_optimizers(total_steps=8)
    metric_name = "train/map_l2" if solver_name == "otp" else "train/endpoint_loss"
    values = []
    for _ in range(8):
        seed_all(999, deterministic=True)
        metrics = solver.training_step(
            easy_ot_batch,
            scaler=disabled_grad_scaler,
            autocast_context=null_autocast,
            gradient_clip_norm=1.0,
        )
        values.append(float(metrics[metric_name]))
    assert statistics.median(values[-3:]) <= statistics.median(values[:3]) + 1.0e-1


@pytest.mark.parametrize("solver_name", ["otp", "flow"])
def test_new_solver_gradients_are_finite_and_nonzero(
    solver_name: str,
    solver_config_factory: object,
    ot_model_config: dict[str, object],
    tiny_training_config: dict[str, object],
    easy_ot_batch: dict[str, torch.Tensor],
    disabled_grad_scaler: torch.amp.GradScaler,
    null_autocast: object,
) -> None:
    solver = build_solver(
        ot_model_config,
        solver_config_factory(solver_name),  # type: ignore[operator]
        tiny_training_config,
    )
    solver.configure_optimizers(total_steps=2)
    for parameter in solver.parameters():
        parameter.grad = None
    solver.training_step(
        easy_ot_batch,
        scaler=disabled_grad_scaler,
        autocast_context=null_autocast,
        gradient_clip_norm=1.0,
    )
    gradients = [parameter.grad for parameter in solver.parameters() if parameter.requires_grad and parameter.grad is not None]
    assert gradients
    assert all(torch.isfinite(gradient).all() for gradient in gradients)
    assert sum(float(gradient.abs().sum()) for gradient in gradients) > 0.0
