from __future__ import annotations

import pytest
import torch

from src.diagnostics.flatness import (
    empirical_semidual_objective,
    make_noisy_potential_copy,
    summarize_flatness,
)
from src.models.potential import PotentialMLP

pytestmark = pytest.mark.unit


def test_empirical_semidual_objective_returns_finite_scalar() -> None:
    potential = PotentialMLP(input_dim=2, hidden_dims=[8, 8], activation="gelu", layer_norm=True)
    source = torch.randn(16, 2)
    target = torch.randn(16, 2)

    value = empirical_semidual_objective(
        lambda x: x + 0.25,
        potential,
        source,
        target,
    )

    assert isinstance(value, float)
    assert torch.isfinite(torch.tensor(value))


def test_noisy_potential_copy_does_not_mutate_original() -> None:
    potential = PotentialMLP(input_dim=2, hidden_dims=[8, 8], activation="gelu", layer_norm=True)
    baseline_parameters = [parameter.detach().clone() for parameter in potential.parameters()]

    noisy_copy = make_noisy_potential_copy(potential, noise_scale=1.0e-2, seed=17)

    assert noisy_copy is not potential
    assert all(
        torch.allclose(before, after)
        for before, after in zip(baseline_parameters, potential.parameters())
    )
    assert any(
        not torch.allclose(original, perturbed)
        for original, perturbed in zip(potential.parameters(), noisy_copy.parameters())
    )


def test_flatness_summary_is_zero_when_all_values_match() -> None:
    summary = summarize_flatness(
        {
            "final": 1.5,
            "early": 1.5,
            "random_init": 1.5,
            "noisy_final": 1.5,
        }
    )

    assert summary["flatness_std_F"] == pytest.approx(0.0, abs=1.0e-8)
    assert summary["flatness_range_F"] == pytest.approx(0.0, abs=1.0e-8)
    assert summary["flatness_mean_abs_gap_to_final"] == pytest.approx(0.0, abs=1.0e-8)
