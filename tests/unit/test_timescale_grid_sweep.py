from __future__ import annotations

from pathlib import Path

import pytest
import torch

import scripts.timescale_grid_sweep as timescale_grid
from scripts.paper_case1_formulation_suite import SYNTHETIC_SOLVER_SPECS
from src.solvers.registry import build_solver


def test_transport_step_sweep_policy() -> None:
    for solver in ["otp", "monge_map", "otm", "maxcorr"]:
        assert timescale_grid._sweep_k_values(solver, [1, 2, 5]) == [1, 2, 5]
        assert timescale_grid._effective_transport_steps(solver, 20) == 20


@pytest.mark.parametrize("solver_name", ["otp", "monge_map", "otm", "maxcorr"])
def test_grid_config_sweeps_transport_steps_inner_steps_and_lr_ratio(solver_name: str) -> None:
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
            "solver.transport_steps=1",
            "solver.inner_steps=1",
            "solver.transport_lr=1.0e-6",
            "solver.potential_lr=1.0e-6",
        ],
    )
    assert config["solver"]["transport_steps"] == 20
    assert config["solver"]["inner_steps"] == 20
    assert config["solver"]["transport_lr"] == 5.0e-4
    assert config["solver"]["potential_lr"] == 5.0e-5
    assert config["dataset"]["seed"] == 0
    assert config["solver"]["noise"]["sigma_start"] == 0.0
    assert config["solver"]["noise"]["sigma_end"] == 0.0


@pytest.mark.parametrize("solver_name", ["otp", "monge_map", "otm", "maxcorr"])
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
        k_value=3,
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
            "solver.transport_steps=1",
            "solver.inner_steps=1",
        ],
    )
    solver = build_solver(config["model"], config["solver"], config["training"])
    solver.configure_optimizers(total_steps=2)
    assert solver.transport_steps == 3
    assert solver.potential_steps == 1
    assert solver.transport_lr == 5.0e-4
    assert solver.potential_lr == pytest.approx(5.0e-5)

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
    assert step_counts[0]["count"] == 3
    assert step_counts[1]["count"] == 1


@pytest.mark.parametrize("solver_name", ["otp", "monge_map", "otm", "maxcorr"])
def test_existing_grid_result_with_runtime_solver_steps_can_be_reused(solver_name: str) -> None:
    result = {
        "solver_id": solver_name,
        "max_steps": 8192,
        "metrics": {
            "solver_transport_steps": 2,
            "solver_potential_steps": 1,
            "solver_transport_lr": 5.0e-4,
            "solver_potential_lr": 5.0e-5,
            "configured_transport_steps": 2,
            "configured_k": 2,
            "configured_transport_lr": 5.0e-4,
            "configured_potential_lr": 5.0e-5,
            "configured_ratio": 0.1,
        },
    }
    assert timescale_grid._existing_matches_grid(
        result,
        solver_name=solver_name,
        max_steps=8192,
        transport_steps=2,
        potential_steps=1,
        transport_lr=5.0e-4,
        potential_lr=5.0e-5,
        ratio_value=0.1,
    )


@pytest.mark.parametrize("solver_name", ["otp", "monge_map", "otm", "maxcorr"])
def test_existing_grid_result_without_runtime_solver_steps_can_be_reused(solver_name: str) -> None:
    legacy_result = {
        "solver_id": solver_name,
        "max_steps": 8192,
        "metrics": {
            "configured_transport_lr": 5.0e-4,
            "configured_potential_lr": 5.0e-5,
            "configured_transport_steps": 2,
            "configured_potential_steps": 1,
            "configured_k": 2,
            "configured_ratio": 0.1,
        },
    }
    assert timescale_grid._existing_matches_grid(
        legacy_result,
        solver_name=solver_name,
        max_steps=8192,
        transport_steps=2,
        potential_steps=1,
        transport_lr=5.0e-4,
        potential_lr=5.0e-5,
        ratio_value=0.1,
    )


def test_grid_config_sets_synthetic_dataset_seed_from_grid_seed() -> None:
    config = timescale_grid._build_grid_config(
        solver_name="otp",
        dataset_name="synthetic_ot_harder",
        seed=48156,
        output_dir=timescale_grid.ROOT / "outputs" / "tmp_timescale_test",
        base_spec=SYNTHETIC_SOLVER_SPECS["otp"],
        cache_version="test",
        device="cpu",
        eval_items=64,
        k_value=2,
        ratio_value=0.1,
        transport_lr=5.0e-4,
        potential_steps=1,
        max_steps=8,
        batch_size=32,
        steps_per_epoch=4,
        disable_noise=True,
        visualize=False,
        save_epoch_checkpoints=False,
        extra_overrides=[],
    )
    assert config["training"]["seed"] == 48156
    assert config["dataset"]["seed"] == 48156


def test_prune_run_checkpoints_keeps_best_only(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    checkpoint_dir = run_dir / "checkpoints"
    checkpoint_dir.mkdir(parents=True)
    for name in ["best.pt", "last.pt", "epoch_0001.pt"]:
        (checkpoint_dir / name).write_text(name, encoding="utf-8")

    timescale_grid._prune_run_checkpoints(
        {"training": {"checkpointing": {"dirpath": "checkpoints"}}},
        run_dir,
    )

    assert sorted(path.name for path in checkpoint_dir.glob("*.pt")) == ["best.pt"]
