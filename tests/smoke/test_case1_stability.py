from __future__ import annotations

from pathlib import Path

import pytest

from src.benchmarking import resolve_ot_dataset, train_baseline_run
from src.diagnostics.case1_stability import run_case1_stability_diagnostic
from src.utils.device import infer_device

pytestmark = pytest.mark.smoke


@pytest.mark.parametrize("solver_name", ["mmv2", "tw2"])
def test_case1_stability_diagnostic_smoke(
    solver_name: str,
    baseline_config_factory: object,
    tmp_output_dir: Path,
) -> None:
    run_dir = tmp_output_dir / f"{solver_name}_case1_stability"
    config = baseline_config_factory(  # type: ignore[operator]
        solver_name,
        output_dir=str(run_dir),
    )
    config["training"]["checkpointing"]["save_every_n_epochs"] = 1
    config["training"]["checkpointing"]["monitor"] = "val/map_l2"
    config["training"]["checkpointing"]["mode"] = "min"
    train_baseline_run(config, output_root=run_dir)

    dataset_bundle = resolve_ot_dataset(config)
    metrics = run_case1_stability_diagnostic(
        run_dir=run_dir,
        config=config,
        dataset_bundle=dataset_bundle,
        device=infer_device(str(config["training"]["device"])),
        max_items=16,
        noise_scale=1.0e-2,
        seed=int(config["training"]["seed"]),
    )

    output_dir = run_dir / "stability_diagnostic"
    assert (run_dir / "checkpoints" / "last.pt").exists()
    assert (output_dir / "stability_results.csv").exists()
    assert (output_dir / "stability_results.json").exists()
    assert (output_dir / "stability_curve.png").exists()
    assert "stability_best_potential_centered_rmse" in metrics
    assert "stability_best_forward_flatness_std_F" in metrics
