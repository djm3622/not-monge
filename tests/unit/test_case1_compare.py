from __future__ import annotations

import scripts.paper_case1_full_compare as case1_compare


def test_case1_compare_defaults_include_otp() -> None:
    assert "otp" in case1_compare.DEFAULT_SOLVERS
    assert "otp" in case1_compare.DEFAULT_FIGURE_SOLVERS
    assert case1_compare.SOLVER_SPECS["otp"] == {
        "max_steps": 1024,
        "batch_size": 512,
        "steps_per_epoch": 64,
        "extra_overrides": [],
    }
