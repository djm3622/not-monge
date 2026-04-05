from __future__ import annotations

import copy
import statistics

import pytest

from src.solvers.registry import build_solver
from src.training.diffusion_task import DiffusionTrainingTask
from src.utils.seed import seed_all

pytestmark = pytest.mark.integration


def _median_trend(values: list[float]) -> tuple[float, float]:
    return statistics.median(values[:3]), statistics.median(values[-3:])


def _trend_training_config(base_config: dict[str, object]) -> dict[str, object]:
    config = copy.deepcopy(base_config)
    config["optimizer"]["lr"] = 5.0e-3
    config["max_steps"] = 8
    return config


def _ot_solver_with_higher_lr(
    solver_name: str,
    ot_model_config: dict[str, object],
    tiny_training_config: dict[str, object],
    solver_config_factory: object,
) -> object:
    training_config = _trend_training_config(tiny_training_config)
    solver_config = solver_config_factory(solver_name)  # type: ignore[operator]
    for key in ["map_lr", "potential_lr", "forward_lr", "inverse_lr", "inner_lr", "lr"]:
        if key in solver_config:
            solver_config[key] = 5.0e-3
    solver = build_solver(ot_model_config, solver_config, training_config)
    solver.configure_optimizers(total_steps=8)
    return solver


@pytest.mark.parametrize("seed", [0, 1])
def test_minimax_map_error_trend_across_seeds(
    seed: int,
    ot_model_config: dict[str, object],
    tiny_training_config: dict[str, object],
    solver_config_factory: object,
    easy_ot_batch: dict[str, object],
    disabled_grad_scaler: object,
    null_autocast: object,
) -> None:
    seed_all(seed, deterministic=True)
    solver = _ot_solver_with_higher_lr("minimax", ot_model_config, tiny_training_config, solver_config_factory)
    losses = []
    for _ in range(8):
        seed_all(999, deterministic=True)
        metrics = solver.training_step(
            easy_ot_batch,
            scaler=disabled_grad_scaler,
            autocast_context=null_autocast,
            gradient_clip_norm=1.0,
        )
        losses.append(float(metrics["train/map_l2"]))
    first, last = _median_trend(losses)
    assert last <= first + 5.0e-2


@pytest.mark.parametrize("solver_name", ["icnn", "mm_b", "entropic"])
def test_representative_ot_solvers_map_error_trend(
    solver_name: str,
    ot_model_config: dict[str, object],
    tiny_training_config: dict[str, object],
    solver_config_factory: object,
    easy_ot_batch: dict[str, object],
    disabled_grad_scaler: object,
    null_autocast: object,
) -> None:
    solver = _ot_solver_with_higher_lr(solver_name, ot_model_config, tiny_training_config, solver_config_factory)
    losses = []
    for _ in range(8):
        seed_all(999, deterministic=True)
        metrics = solver.training_step(
            easy_ot_batch,
            scaler=disabled_grad_scaler,
            autocast_context=null_autocast,
            gradient_clip_norm=1.0,
        )
        losses.append(float(metrics["train/map_l2"]))
    first, last = _median_trend(losses)
    assert last <= first + 5.0e-2


@pytest.mark.parametrize("seed", [0, 1])
def test_diffusion_loss_trend_across_seeds(
    seed: int,
    tiny_diffusion_model_config: dict[str, object],
    tiny_training_config: dict[str, object],
    diffusion_batch: dict[str, object],
    disabled_grad_scaler: object,
    null_autocast: object,
) -> None:
    seed_all(seed, deterministic=True)
    training_config = _trend_training_config(tiny_training_config)
    task = DiffusionTrainingTask(
        model_config=copy.deepcopy(tiny_diffusion_model_config),
        training_config=training_config,
    )
    task.configure_optimizers(total_steps=8)
    losses = []
    for _ in range(8):
        seed_all(999, deterministic=True)
        metrics = task.training_step(
            diffusion_batch,
            scaler=disabled_grad_scaler,
            autocast_context=null_autocast,
            gradient_clip_norm=1.0,
        )
        losses.append(float(metrics["train/ddpm_loss"]))
    first, last = _median_trend(losses)
    assert last <= first + 1.0e-1
