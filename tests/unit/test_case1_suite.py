from __future__ import annotations

import scripts.paper_case1_suite as case1_suite


def test_case1_suite_defaults_include_published_direct_map_formulations() -> None:
    assert case1_suite.DEFAULT_SOLVERS == [
        "gaussian",
        "mm",
        "mmv2",
        "tw2",
        "mm_b",
        "qc",
        "otp",
        "monge_map",
        "otm",
        "maxcorr",
        "makkuva_icnn_cvx",
    ]
    assert case1_suite.SOLVER_SPECS["otp"] == {
        "max_steps": 4096,
        "batch_size": 256,
        "steps_per_epoch": 64,
        "extra_overrides": [
            "solver.transport_steps=1",
            "solver.transport_lr=5e-4",
            "solver.potential_lr=5e-4",
            "solver.noise.sigma_start=0.0",
            "solver.noise.sigma_end=0.0",
        ],
    }
    assert case1_suite.SOLVER_SPECS["otm"]["max_steps"] == 4096
    assert "otm" in case1_suite.SOLVER_SPECS
    assert "monge_map" in case1_suite.SOLVER_SPECS


def test_case1_suite_table_names_match_requested_labels() -> None:
    assert case1_suite.LATEX_SOLVER_NAMES["mm"] == "tMM"
    assert case1_suite.LATEX_SOLVER_NAMES["mmv2"] == "tMMv2"
    assert case1_suite.LATEX_SOLVER_NAMES["tw2"] == "tW2"
    assert case1_suite.LATEX_SOLVER_NAMES["otm"] == "OTM"


def test_case1_suite_stability_defaults_target_learned_solvers_only() -> None:
    assert case1_suite.DEFAULT_STABILITY_CHECKPOINTS == 5
    assert case1_suite.CASE1_STABILITY_SOLVERS == {"mm", "mmv2", "tw2", "mm_b", "qc"}
    assert not case1_suite._supports_case1_stability("gaussian")


def test_case1_suite_stability_checkpoint_interval_spreads_saved_epochs() -> None:
    assert case1_suite._stability_checkpoint_interval(max_steps=4096, steps_per_epoch=128, num_checkpoints=5) == 6
    assert case1_suite._stability_checkpoint_interval(max_steps=10000, steps_per_epoch=128, num_checkpoints=5) == 15
    assert case1_suite._stability_checkpoint_interval(max_steps=1024, steps_per_epoch=64, num_checkpoints=0) == 0


def test_case1_suite_budget_scale_keeps_gaussian_fixed() -> None:
    scaled = case1_suite._scale_solver_specs(case1_suite.SOLVER_SPECS, budget_scale=2.0)
    assert scaled["gaussian"]["max_steps"] == 1
    assert scaled["otm"]["max_steps"] == 8192
