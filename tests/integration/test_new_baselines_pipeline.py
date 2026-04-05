from __future__ import annotations

import copy
from pathlib import Path

import pytest

from src.benchmarking import eval_baseline_run, train_baseline_run

pytestmark = pytest.mark.integration


@pytest.mark.parametrize("solver_name", ["otp", "flow"])
def test_new_baselines_train_and_eval_pipeline(
    solver_name: str,
    baseline_config_factory: object,
    tmp_output_dir: Path,
) -> None:
    train_root = tmp_output_dir / solver_name
    config = baseline_config_factory(solver_name, output_dir=f"outputs/{solver_name}")  # type: ignore[operator]
    train_result = train_baseline_run(config, output_root=train_root)
    checkpoint_path = train_root / "checkpoints" / "best.pt"
    assert train_result["solver_id"] == solver_name
    assert checkpoint_path.exists()
    assert (train_root / "results.json").exists()
    assert {"map_l2", "pushforward_w2", "mmd"} <= set(train_result["metrics"])

    eval_config = copy.deepcopy(config)
    eval_config["evaluation"]["checkpoint_path"] = str(checkpoint_path)
    eval_root = tmp_output_dir / f"{solver_name}_eval"
    eval_result = eval_baseline_run(
        eval_config,
        checkpoint_path=checkpoint_path,
        output_root=eval_root,
    )
    assert eval_result["solver_id"] == solver_name
    assert (eval_root / "results.json").exists()
    assert {"map_l2", "pushforward_w2", "mmd"} <= set(eval_result["metrics"])
