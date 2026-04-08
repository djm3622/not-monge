from __future__ import annotations

import scripts.paper_case1_suite as case1_suite


def test_case1_suite_defaults_exclude_otp_from_paper_table() -> None:
    assert case1_suite.DEFAULT_SOLVERS == ["gaussian", "mm", "mmv2", "tw2", "mm_b", "qc"]
    assert "otp" not in case1_suite.DEFAULT_SOLVERS
    assert case1_suite.SOLVER_SPECS["otp"] == {
        "max_steps": 1024,
        "batch_size": 512,
        "steps_per_epoch": 64,
        "extra_overrides": [],
    }


def test_case1_suite_table_names_match_requested_labels() -> None:
    assert case1_suite.LATEX_SOLVER_NAMES["mm"] == "tMM"
    assert case1_suite.LATEX_SOLVER_NAMES["mmv2"] == "tMMv2"
    assert case1_suite.LATEX_SOLVER_NAMES["tw2"] == "tW2"
