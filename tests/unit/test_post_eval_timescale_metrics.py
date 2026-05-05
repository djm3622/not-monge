from __future__ import annotations

import json
from pathlib import Path

import pytest

import scripts.post_eval_timescale_metrics as post_eval


def _write_run(
    root: Path,
    *,
    solver: str,
    k_value: int,
    ratio_slug: str,
    seed: int,
    final_map_l2: float,
    final_flatness: float,
) -> Path:
    run_dir = root / solver / f"k{k_value}_r{ratio_slug}" / f"seed{seed}"
    run_dir.mkdir(parents=True)
    result = {
        "solver_id": solver,
        "dataset_id": "synthetic_ot_harder",
        "seed": seed,
        "max_steps": 4,
        "batch_size": 8,
        "metrics": {
            "configured_k": k_value,
            "configured_ratio": float(ratio_slug.replace("p", ".")),
            "configured_transport_steps": k_value,
            "configured_potential_steps": 1,
            "configured_transport_lr": 5.0e-4,
            "configured_potential_lr": 5.0e-5,
        },
    }
    (run_dir / "results.json").write_text(json.dumps(result), encoding="utf-8")
    rows = [
        {"step": 1, "val/map_l2": 99.0, "val/flatness_std_F": final_flatness + 10.0},
        {"step": 4, "val/map_l2": final_map_l2, "val/flatness_std_F": final_flatness},
    ]
    (run_dir / "metrics.jsonl").write_text(
        "\n".join(json.dumps(row) for row in rows) + "\n",
        encoding="utf-8",
    )
    return run_dir / "metrics.jsonl"


def test_collect_final_metric_rows_and_aggregate_seed_stats(tmp_path: Path) -> None:
    metrics_files = [
        _write_run(
            tmp_path,
            solver="otp",
            k_value=2,
            ratio_slug="0p1",
            seed=0,
            final_map_l2=1.0,
            final_flatness=3.0,
        ),
        _write_run(
            tmp_path,
            solver="otp",
            k_value=2,
            ratio_slug="0p1",
            seed=1,
            final_map_l2=3.0,
            final_flatness=7.0,
        ),
    ]
    metric_names = ["val/map_l2", "val/flatness_std_F"]

    final_rows, best_rows = post_eval.collect_final_metric_rows(metrics_files, metric_names)
    summary = post_eval.aggregate_final_rows(final_rows, metric_names)

    assert len(final_rows) == 2
    assert {row["final_step"] for row in final_rows} == {4}
    assert len(summary) == 1
    assert summary[0]["val_map_l2_mean"] == pytest.approx(2.0)
    assert summary[0]["val_map_l2_std"] == pytest.approx(1.0)
    assert summary[0]["val_flatness_std_F_mean"] == pytest.approx(5.0)
    assert summary[0]["val_flatness_std_F_std"] == pytest.approx(2.0)

    map_best = [row for row in best_rows if row["metric"] == "val/map_l2"]
    assert len(map_best) == 2
    assert {row["best_value"] for row in map_best} == {1.0, 3.0}
