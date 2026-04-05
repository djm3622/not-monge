from __future__ import annotations

import copy
from pathlib import Path

import pytest

from src.benchmarking import eval_baseline_run, generate_tables_from_results, train_baseline_run

pytestmark = pytest.mark.integration


def test_ot_pipeline_train_eval_and_table_generation(
    baseline_config_factory: object,
    tmp_output_dir: Path,
) -> None:
    train_root = tmp_output_dir / "minimax"
    config = baseline_config_factory("minimax", output_dir="outputs/minimax")  # type: ignore[operator]
    train_result = train_baseline_run(config, output_root=train_root)
    checkpoint_path = train_root / "checkpoints" / "best.pt"

    assert train_result["solver_id"] == "minimax"
    assert checkpoint_path.exists()
    assert (train_root / "results.json").exists()
    assert {"map_l2", "pushforward_w2", "mmd"} <= set(train_result["metrics"])

    eval_config = copy.deepcopy(config)
    eval_config["evaluation"]["checkpoint_path"] = str(checkpoint_path)
    eval_root = tmp_output_dir / "minimax_eval"
    eval_result = eval_baseline_run(
        eval_config,
        checkpoint_path=checkpoint_path,
        output_root=eval_root,
    )
    assert eval_result["solver_id"] == "minimax"
    assert {"map_l2", "pushforward_w2", "mmd"} <= set(eval_result["metrics"])
    assert (eval_root / "results.json").exists()

    grouped = generate_tables_from_results(tmp_output_dir, tmp_output_dir / "tables")
    assert "ot_recovery" in grouped
    assert (tmp_output_dir / "tables" / "ot_recovery.csv").exists()
    assert (tmp_output_dir / "tables" / "ot_recovery.tex").exists()


def test_c_concavity_pipeline_emits_convexity_metrics(
    baseline_config_factory: object,
    tmp_output_dir: Path,
) -> None:
    train_root = tmp_output_dir / "icnn_concavity"
    config = baseline_config_factory(  # type: ignore[operator]
        "icnn",
        output_dir="outputs/icnn_concavity",
        experiment_id="c_concavity",
    )
    result = train_baseline_run(config, output_root=train_root)
    metrics = result["metrics"]
    assert "envelope_gap/mean" in metrics
    assert "convexity_violation/mean" in metrics
    assert "hessian/min_eig" in metrics
