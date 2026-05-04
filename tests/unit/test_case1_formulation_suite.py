from __future__ import annotations

import scripts.paper_case1_formulation_suite as formulation_suite


def test_formulation_suite_defaults_cover_published_direct_map_families() -> None:
    assert formulation_suite.DEFAULT_SOLVERS == ["otp", "monge_map", "otm", "maxcorr", "makkuva_icnn_cvx"]
    assert set(formulation_suite.SOLVER_SPECS) == {"otp", "monge_map", "otm", "maxcorr", "makkuva_icnn_cvx"}
    assert set(formulation_suite.SYNTHETIC_SOLVER_SPECS) == {"otp", "monge_map", "otm", "maxcorr", "makkuva_icnn_cvx"}
    assert formulation_suite.DEFAULT_DIAGNOSTIC_CHECKPOINTS == 5
    assert formulation_suite.DEFAULT_DIAGNOSTIC_ITEMS == 512
    assert formulation_suite.SOLVER_SPECS["otm"]["max_steps"] == 4096


def test_formulation_suite_direct_map_solvers_pin_sweep_timescale_overrides() -> None:
    expected = {
        "solver.transport_steps=1",
        "solver.transport_lr=5e-4",
        "solver.potential_lr=5e-4",
        "solver.noise.sigma_start=0.0",
        "solver.noise.sigma_end=0.0",
    }
    for specs in [formulation_suite.SOLVER_SPECS, formulation_suite.SYNTHETIC_SOLVER_SPECS]:
        for solver in ["monge_map", "otm", "maxcorr"]:
            assert expected.issubset(set(specs[solver]["extra_overrides"]))


def test_formulation_suite_budget_scale_extends_all_direct_rows() -> None:
    scaled = formulation_suite._scale_solver_specs(formulation_suite.SOLVER_SPECS, budget_scale=4.0)
    assert scaled["otp"]["max_steps"] == 16384
    assert scaled["otm"]["max_steps"] == 16384
    assert scaled["makkuva_icnn_cvx"]["max_steps"] == 16384


def test_build_run_config_uses_synthetic_dataset_dimension() -> None:
    config = formulation_suite._build_run_config(
        solver_name="otp",
        dataset_name="synthetic_ot",
        seed=686499,
        output_dir=formulation_suite.ROOT / "outputs" / "tmp_test_suite",
        spec=formulation_suite.SYNTHETIC_SOLVER_SPECS["otp"],
        cache_version=formulation_suite.DEFAULT_CACHE_VERSION,
        device="cpu",
        eval_items=256,
        visualization_items=64,
        saddle_examples=2,
        diagnostic_checkpoints=2,
        overrides=[],
    )
    assert config["dataset"]["name"] == "synthetic_ot"
    assert config["model"]["input_dim"] == 8
    assert config["model"]["output_dim"] == 8
    assert config["dataset"]["batch_size"] == 512
