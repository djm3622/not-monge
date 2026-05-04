from __future__ import annotations

import pytest
import torch

import scripts.timescale_grid_sweep as timescale_grid
from scripts.paper_case1_formulation_suite import SYNTHETIC_SOLVER_SPECS
from src.solvers.registry import build_solver


def test_direct_map_solvers_use_fixed_transport_steps_in_grid() -> None:
    assert timescale_grid._sweep_k_values("otp", [1, 2, 5]) == [1, 2, 5]
    for solver in ["monge_map", "otm", "maxcorr"]:
        assert timescale_grid._sweep_k_values(solver, [1, 2, 5]) == [1]
        assert timescale_grid._effective_transport_steps(solver, 20) == 1


@pytest.mark.parametrize("solver_name", ["monge_map", "otm", "maxcorr"])
def test_grid_config_pins_direct_map_transport_steps_and_lr_ratio(solver_name: str) -> None:
    config = timescale_grid._build_grid_config(
        solver_name=solver_name,
        dataset_name="synthetic_ot",
        seed=0,
        output_dir=timescale_grid.ROOT / "outputs" / "tmp_timescale_test",
        base_spec=SYNTHETIC_SOLVER_SPECS[solver_name],
        cache_version="test",
        device="cpu",
        eval_items=64,
        k_value=20,
        ratio_value=0.1,
        transport_lr=5.0e-4,
        potential_steps=1,
        max_steps=8,
        batch_size=32,
        steps_per_epoch=4,
        disable_noise=True,
        visualize=False,
        save_epoch_checkpoints=False,
        extra_overrides=[
            "solver.transport_steps=20",
            "solver.transport_lr=1.0e-6",
            "solver.potential_lr=1.0e-6",
        ],
    )
    assert config["solver"]["transport_steps"] == 1
    assert config["solver"]["transport_lr"] == 5.0e-4
    assert config["solver"]["potential_lr"] == 5.0e-5
    assert config["solver"]["noise"]["sigma_start"] == 0.0
    assert config["solver"]["noise"]["sigma_end"] == 0.0


@pytest.mark.parametrize("solver_name", ["monge_map", "otm", "maxcorr"])
def test_grid_config_controls_actual_solver_optimizer_step_count(
    solver_name: str,
    disabled_grad_scaler: torch.amp.GradScaler,
    null_autocast: object,
) -> None:
    config = timescale_grid._build_grid_config(
        solver_name=solver_name,
        dataset_name="synthetic_ot",
        seed=0,
        output_dir=timescale_grid.ROOT / "outputs" / "tmp_timescale_test",
        base_spec=SYNTHETIC_SOLVER_SPECS[solver_name],
        cache_version="test",
        device="cpu",
        eval_items=64,
        k_value=20,
        ratio_value=0.1,
        transport_lr=5.0e-4,
        potential_steps=1,
        max_steps=8,
        batch_size=32,
        steps_per_epoch=4,
        disable_noise=True,
        visualize=False,
        save_epoch_checkpoints=False,
        extra_overrides=[
            "model.hidden_dims=[8,8]",
            "model.residual=false",
            "model.layer_norm=false",
            "solver.potential.hidden_dims=[8,8]",
            "solver.transport_steps=20",
        ],
    )
    solver = build_solver(config["model"], config["solver"], config["training"])
    solver.configure_optimizers(total_steps=2)
    assert solver.transport_steps == 1
    assert solver.potential_steps == 1
    assert solver.transport_lr == 5.0e-4
    assert solver.potential_lr == 5.0e-5

    step_counts = []
    for optimizer in solver.optimizers:
        counter = {"count": 0}
        original_step = optimizer.step

        def counted_step(
            *args: object,
            _counter: dict[str, int] = counter,
            _original_step: object = original_step,
            **kwargs: object,
        ) -> object:
            _counter["count"] += 1
            return _original_step(*args, **kwargs)  # type: ignore[operator]

        optimizer.step = counted_step  # type: ignore[method-assign]
        step_counts.append(counter)

    source = torch.randn(4, 8)
    target = torch.randn(4, 8)
    solver.training_step(
        {"source": source, "target": target, "ground_truth_map": target},
        scaler=disabled_grad_scaler,
        autocast_context=null_autocast,
        gradient_clip_norm=1.0,
    )
    assert step_counts[0]["count"] == 1
    assert step_counts[1]["count"] == 1


def test_existing_grid_result_without_runtime_solver_steps_is_stale() -> None:
    stale_result = {
        "solver_id": "monge_map",
        "max_steps": 8192,
        "metrics": {
            "configured_transport_steps": 1,
            "configured_k": 1,
            "configured_transport_lr": 5.0e-4,
            "configured_potential_lr": 1.0e-5,
            "configured_ratio": 0.02,
        },
    }
    assert not timescale_grid._existing_matches_grid(
        stale_result,
        solver_name="monge_map",
        max_steps=8192,
        transport_steps=1,
        potential_steps=1,
        transport_lr=5.0e-4,
        potential_lr=1.0e-5,
        ratio_value=0.02,
    )


def test_existing_grid_result_with_runtime_solver_steps_can_be_reused() -> None:
    result = {
        "solver_id": "monge_map",
        "max_steps": 8192,
        "metrics": {
            "solver_transport_steps": 1,
            "solver_potential_steps": 1,
            "solver_transport_lr": 5.0e-4,
            "solver_potential_lr": 1.0e-5,
            "configured_transport_steps": 1,
            "configured_k": 1,
            "configured_transport_lr": 5.0e-4,
            "configured_potential_lr": 1.0e-5,
            "configured_ratio": 0.02,
        },
    }
    assert timescale_grid._existing_matches_grid(
        result,
        solver_name="monge_map",
        max_steps=8192,
        transport_steps=1,
        potential_steps=1,
        transport_lr=5.0e-4,
        potential_lr=1.0e-5,
        ratio_value=0.02,
    )
