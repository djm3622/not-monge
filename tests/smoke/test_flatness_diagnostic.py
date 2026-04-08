from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest

from src.benchmarking import train_baseline_run

pytestmark = pytest.mark.smoke


@pytest.mark.parametrize("solver_name", ["makkuva_icnn_cvx", "makkuva_mlp_ablation"])
def test_flatness_diagnostic_script_smoke(
    solver_name: str,
    baseline_config_factory: object,
    tiny_makkuva_checkerboard_config: dict[str, object],
    tmp_output_dir: Path,
) -> None:
    run_dir = tmp_output_dir / f"{solver_name}_flatness"
    config = baseline_config_factory(  # type: ignore[operator]
        solver_name,
        output_dir=str(run_dir),
        dataset_config=tiny_makkuva_checkerboard_config,
    )
    config["training"]["checkpointing"]["monitor"] = "val/pushforward_w2"
    config["training"]["checkpointing"]["mode"] = "min"
    train_baseline_run(config, output_root=run_dir)

    command = [
        sys.executable,
        "scripts/flatness_diagnostic.py",
        "--run-dir",
        str(run_dir),
        "--num-checkpoints",
        "1",
        "--max-items",
        "16",
        "--override",
        "training.device=cpu",
        "--override",
        "dataset.seed=7",
        "--override",
        "dataset.steps_per_epoch=2",
        "--override",
        "dataset.n_val=16",
        "--override",
        "dataset.n_test=16",
        "--override",
        "dataset.standardize_samples=64",
    ]
    subprocess.run(
        command,
        cwd=Path(__file__).resolve().parents[2],
        check=True,
        env={**os.environ, "PYTHONPYCACHEPREFIX": "/tmp/pycache"},
    )

    output_dir = run_dir / "flatness_diagnostic"
    assert (output_dir / "flatness_results.csv").exists()
    assert (output_dir / "flatness_vs_step.png").exists()
    assert (output_dir / "flatness_vs_pushforward_w2.png").exists()
