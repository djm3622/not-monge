"""Reference target potentials must match each solver's own (scaled/regularised) objective.

Uses a quadratic Brenier potential Phi(x) = 1/2 x^T A x, so T*(x) = A x and
Phi*(y) = 1/2 y^T A^{-1} y are available in closed form.
"""

from __future__ import annotations

import pytest
import torch

from src.solvers.registry import build_solver
from src.training.losses import quadratic_cost

pytestmark = pytest.mark.unit

_SOLVER_OVERRIDES = [
    ("otp", {}),
    ("otp", {"quadratic_scale": 0.015625}),
    ("monge_map", {"quadratic_scale": 0.015625}),
    ("monge_map", {"quadratic_scale": 0.5}),
    ("maxcorr", {"transport_l2_weight": 0.05}),
    ("maxcorr", {"transport_l2_weight": 0.0, "dot_reward_scale": 1.0}),
    ("otm", {"dot_reward_scale": 0.015625, "transport_l2_weight": 0.0}),
    ("otm", {"dot_reward_scale": 0.5, "transport_l2_weight": 0.1}),
]


def _quadratic_problem() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    generator = torch.Generator().manual_seed(0)
    basis = torch.randn(2, 2, generator=generator, dtype=torch.float64)
    matrix = basis @ basis.T + 0.5 * torch.eye(2, dtype=torch.float64)
    source = torch.randn(64, 2, generator=generator, dtype=torch.float64)
    target = source @ matrix.T
    source_values = 0.5 * (source @ matrix * source).sum(dim=-1)
    return matrix, source, target, source_values


def _build(solver_name, overrides, solver_config_factory, ot_model_config, tiny_training_config):
    config = solver_config_factory(solver_name)  # type: ignore[operator]
    config.update(overrides)
    return build_solver(ot_model_config, config, tiny_training_config)


def _closed_form_reference(solver, matrix: torch.Tensor):
    """Return f*(y) built only from the solver's map objective and Phi*."""
    inverse = torch.linalg.inv(matrix)

    def conjugate(y: torch.Tensor) -> torch.Tensor:
        return 0.5 * (y @ inverse * y).sum(dim=-1)

    if solver.solver_name in {"otp", "monge_map"}:
        scale = solver.quadratic_scale
        return lambda y: scale * y.pow(2).sum(dim=-1) - 2.0 * scale * conjugate(y)
    weight = solver.transport_l2_weight
    scale = solver.dot_reward_scale + weight
    return lambda y: scale * conjugate(y) - 0.5 * weight * y.pow(2).sum(dim=-1)


def _map_step_loss(solver, source: torch.Tensor, transported: torch.Tensor, potential_values: torch.Tensor):
    """Per-sample map loss exactly as the solvers' training_step writes it."""
    if solver.solver_name in {"otp", "monge_map"}:
        return solver._transport_cost(source, transported) - potential_values
    dot_reward = solver.dot_reward_scale * (source * transported).sum(dim=-1)
    regularizer = solver.transport_l2_weight * quadratic_cost(source, transported)
    return potential_values - dot_reward + regularizer


@pytest.mark.parametrize(("solver_name", "overrides"), _SOLVER_OVERRIDES)
def test_reference_potential_matches_closed_form(
    solver_name, overrides, solver_config_factory, ot_model_config, tiny_training_config
) -> None:
    solver = _build(solver_name, overrides, solver_config_factory, ot_model_config, tiny_training_config)
    matrix, source, target, source_values = _quadratic_problem()

    values, gradients = solver.reference_target_potential(source, target, source_values)

    target_eval = target.clone().requires_grad_(True)
    expected_values = _closed_form_reference(solver, matrix)(target_eval)
    (expected_gradients,) = torch.autograd.grad(expected_values.sum(), target_eval)
    assert torch.allclose(values, expected_values.detach(), atol=1e-10)
    assert torch.allclose(gradients, expected_gradients, atol=1e-10)


@pytest.mark.parametrize(("solver_name", "overrides"), _SOLVER_OVERRIDES)
def test_reference_potential_makes_true_map_stationary(
    solver_name, overrides, solver_config_factory, ot_model_config, tiny_training_config
) -> None:
    solver = _build(solver_name, overrides, solver_config_factory, ot_model_config, tiny_training_config)
    matrix, source, target, _ = _quadratic_problem()
    reference = _closed_form_reference(solver, matrix)

    transported = target.clone().requires_grad_(True)
    loss = _map_step_loss(solver, source, transported, reference(transported)).sum()
    (map_gradient,) = torch.autograd.grad(loss, transported)
    assert torch.allclose(map_gradient, torch.zeros_like(map_gradient), atol=1e-10)


@pytest.mark.parametrize(("solver_name", "overrides"), _SOLVER_OVERRIDES)
def test_unit_scale_potential_gap_recovers_unit_reference(
    solver_name, overrides, solver_config_factory, ot_model_config, tiny_training_config
) -> None:
    solver = _build(solver_name, overrides, solver_config_factory, ot_model_config, tiny_training_config)
    matrix, source, target, _ = _quadratic_problem()
    prediction = 0.9 * target + 0.1
    reference = _closed_form_reference(solver, matrix)
    inverse = torch.linalg.inv(matrix)

    def unit_potential(y: torch.Tensor) -> torch.Tensor:
        conjugate = 0.5 * (y @ inverse * y).sum(dim=-1)
        if solver.solver_name in {"otp", "monge_map"}:
            return 0.5 * y.pow(2).sum(dim=-1) - conjugate
        return conjugate

    gap = reference(target).mean() - reference(prediction).mean()
    unit_gap = unit_potential(target).mean() - unit_potential(prediction).mean()
    converted = solver.unit_scale_potential_gap(gap, prediction, target)
    assert torch.allclose(converted, unit_gap, atol=1e-10)
