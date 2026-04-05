from __future__ import annotations

from pathlib import Path

import pytest

from src.benchmarking import train_baseline_run

pytestmark = pytest.mark.smoke


@pytest.mark.parametrize(
    ("experiment_id", "solver_name"),
    [
        ("ot_recovery", "minimax"),
        ("ot_recovery", "otp"),
        ("ot_recovery", "flow"),
        ("c_concavity", "icnn"),
        ("c_concavity", "otp"),
        ("diffusion_latent", "minimax"),
        ("diffusion_latent", "otp"),
        ("diffusion_latent", "flow"),
    ],
)
def test_all_experiments_smoke(
    experiment_id: str,
    solver_name: str,
    baseline_config_factory: object,
    tiny_diffusion_latent_dataset_config: dict[str, object],
    fake_timm: object,
    tmp_output_dir: Path,
) -> None:
    del fake_timm
    dataset_config = None
    if experiment_id == "diffusion_latent":
        dataset_config = tiny_diffusion_latent_dataset_config
    config = baseline_config_factory(  # type: ignore[operator]
        solver_name,
        output_dir=f"outputs/{experiment_id}",
        experiment_id=experiment_id,
        dataset_config=dataset_config,
    )
    result = train_baseline_run(config, output_root=tmp_output_dir / experiment_id)
    assert (tmp_output_dir / experiment_id / "results.json").exists()
    if experiment_id == "ot_recovery":
        assert {"map_l2", "pushforward_w2", "mmd"} <= set(result["metrics"])
    elif experiment_id == "c_concavity":
        assert "envelope_gap/mean" in result["metrics"]
        assert "convexity_violation/mean" in result["metrics"]
    else:
        assert "fid" in result["metrics"]
        assert "precision" in result["metrics"]
        assert "recall" in result["metrics"]
        assert result["metrics"]["metric_space"] == "latent"
