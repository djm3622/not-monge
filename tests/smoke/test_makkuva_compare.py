from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

pytestmark = pytest.mark.smoke


def test_makkuva_compare_script_smoke(
    tmp_output_dir: Path,
) -> None:
    output_root = tmp_output_dir / "makkuva_compare"
    command = [
        sys.executable,
        "scripts/makkuva_case1_compare.py",
        "--output-root",
        str(output_root),
        "--seeds",
        "1",
        "--eval-items",
        "16",
        "--override",
        "training.max_steps=2",
        "--override",
        "training.log_every_n_steps=1",
        "--override",
        "dataset.batch_size=8",
        "--override",
        "dataset.steps_per_epoch=2",
        "--override",
        "dataset.n_val=16",
        "--override",
        "dataset.n_test=16",
        "--override",
        "dataset.standardize_samples=32",
    ]
    subprocess.run(command, cwd=Path(__file__).resolve().parents[2], check=True)
    assert (output_root / "summary.json").exists()
    assert (output_root / "summary.csv").exists()
    assert (output_root / "summary.md").exists()
