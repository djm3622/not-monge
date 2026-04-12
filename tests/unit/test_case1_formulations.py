from __future__ import annotations

import pytest
import torch

from src.diagnostics.case1_formulations import (
    extract_potential_module,
    potential_gradient,
    target_reference_potential,
)
from src.solvers.registry import build_solver

pytestmark = pytest.mark.unit


@pytest.mark.parametrize(
    ("solver_name", "expected_symbol"),
    [("otp", "psi"), ("maxcorr", "g")],
)
def test_target_reference_potential_matches_solver_parameterization(
    solver_name: str,
    expected_symbol: str,
    solver_config_factory: object,
    ot_model_config: dict[str, object],
    tiny_training_config: dict[str, object],
    ot_batch: dict[str, torch.Tensor],
    synthetic_bundle: object,
) -> None:
    solver = build_solver(
        ot_model_config,
        solver_config_factory(solver_name),  # type: ignore[operator]
        tiny_training_config,
    )
    source = ot_batch["source"]
    target = ot_batch["ground_truth_map"]
    reference_values, reference_gradients, symbol = target_reference_potential(
        solver,
        synthetic_bundle.ground_truth_potential,
        source=source,
        target=target,
    )
    source_values = synthetic_bundle.ground_truth_potential(source).view(-1)
    conjugate = (source * target).sum(dim=-1) - source_values

    assert symbol == expected_symbol
    if solver_name == "otp":
        expected_values = solver.quadratic_scale * target.pow(2).sum(dim=-1) - conjugate
        expected_gradients = 2.0 * solver.quadratic_scale * target - source
    else:
        expected_values = conjugate
        expected_gradients = source

    assert torch.allclose(reference_values, expected_values)
    assert torch.allclose(reference_gradients, expected_gradients)


def test_extract_potential_module_supports_gradient_for_direct_solver(
    solver_config_factory: object,
    ot_model_config: dict[str, object],
    tiny_training_config: dict[str, object],
    ot_batch: dict[str, torch.Tensor],
) -> None:
    solver = build_solver(
        ot_model_config,
        solver_config_factory("otp"),  # type: ignore[operator]
        tiny_training_config,
    )
    module = extract_potential_module(solver)
    gradients = potential_gradient(module, ot_batch["target"], create_graph=False)
    assert gradients.shape == ot_batch["target"].shape
    assert torch.isfinite(gradients).all()
