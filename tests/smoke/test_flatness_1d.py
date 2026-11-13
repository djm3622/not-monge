from __future__ import annotations

import math

import pytest

from src.diagnostics.flatness_1d import Gaussian1DProblem, run_flatness_1d

pytestmark = pytest.mark.smoke

TINY_CONFIG = {
    "problem": {"mean": 2.0, "std": 1.5},
    "model": {"width": 8},
    "training": {
        "map_lr": 1.0e-3,
        "potential_steps": 20,
        "batch_size": 16,
        "perturb_at": 10,
        "perturb_std": 0.5,
        "freeze_map_after_perturb": False,
    },
    "validation": {"seed": 12345, "num_samples": 64, "grid_points": 11, "log_every": 5},
}


def test_optimal_value_matches_closed_form() -> None:
    assert Gaussian1DProblem().optimal_value == pytest.approx(2.125)


def test_run_is_seeded_and_finite() -> None:
    first = run_flatness_1d(TINY_CONFIG, inner_steps=2, lr_ratio=0.1, seed=0)
    second = run_flatness_1d(TINY_CONFIG, inner_steps=2, lr_ratio=0.1, seed=0)
    assert first == second
    assert first["step"] == [0, 5, 10, 15, 20]
    assert first["T_updates"][-1] == 40
    assert all(math.isfinite(value) for values in first.values() for value in values)
