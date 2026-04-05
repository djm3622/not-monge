from __future__ import annotations

import pytest
import torch

from src.solvers.registry import build_solver

pytestmark = pytest.mark.unit


@pytest.mark.parametrize("solver_name", ["otp", "flow"])
def test_new_solvers_build_and_compute_map(
    solver_name: str,
    solver_config_factory: object,
    ot_model_config: dict[str, object],
    tiny_training_config: dict[str, object],
    ot_batch: dict[str, torch.Tensor],
) -> None:
    solver = build_solver(
        ot_model_config,
        solver_config_factory(solver_name),  # type: ignore[operator]
        tiny_training_config,
    )
    mapped = solver.compute_map(ot_batch["source"])
    assert mapped.shape == ot_batch["source"].shape
    assert torch.isfinite(mapped).all()


def test_otp_compute_loss_and_regularization(
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
    losses = solver.compute_loss(ot_batch, smoothing_sigma=0.02)
    regs = solver.regularization(
        batch={"source": losses["source"], "target": losses["target"]},
        transported=losses["transported"],
        potential_real=losses["potential_real"],
        potential_fake=losses["potential_fake"],
        plan=losses["plan"],
    )
    for key in ["critic_objective", "map_objective", "plan_supervision", "potential_gp"]:
        value = losses[key] if key in losses else regs[key]
        assert torch.isfinite(value).all()
    assert losses["transported"].shape == ot_batch["source"].shape


def test_otp_backward_produces_finite_gradients(
    solver_config_factory: object,
    ot_model_config: dict[str, object],
    tiny_training_config: dict[str, object],
    easy_ot_batch: dict[str, torch.Tensor],
) -> None:
    solver = build_solver(
        ot_model_config,
        solver_config_factory("otp"),  # type: ignore[operator]
        tiny_training_config,
    )
    solver.configure_optimizers(total_steps=2)
    losses = solver.compute_loss(easy_ot_batch, smoothing_sigma=0.0)
    regs = solver.regularization(
        batch={"source": losses["source"], "target": losses["target"]},
        transported=losses["transported"],
        potential_real=losses["potential_real"],
        potential_fake=losses["potential_fake"],
        plan=losses["plan"],
    )
    objective = (
        losses["map_objective"]
        + solver.plan_supervision_weight * regs["plan_supervision"]
        + solver.plan_entropy_weight * regs["negative_entropy"]
    )
    objective.backward()
    gradients = [parameter.grad for parameter in solver.parameters() if parameter.requires_grad and parameter.grad is not None]
    assert gradients
    assert all(torch.isfinite(gradient).all() for gradient in gradients)


def test_flow_forward_flow_and_integrate_flow(
    solver_config_factory: object,
    ot_model_config: dict[str, object],
    tiny_training_config: dict[str, object],
    ot_batch: dict[str, torch.Tensor],
) -> None:
    solver = build_solver(
        ot_model_config,
        solver_config_factory("flow"),  # type: ignore[operator]
        tiny_training_config,
    )
    times = torch.linspace(0.0, 1.0, ot_batch["source"].shape[0])
    velocity = solver.forward_flow(ot_batch["source"], times)
    terminal, trajectory = solver.integrate_flow(ot_batch["source"], return_trajectory=True)
    assert velocity.shape == ot_batch["source"].shape
    assert terminal.shape == ot_batch["source"].shape
    assert trajectory.shape[1:] == ot_batch["source"].shape
    assert torch.isfinite(trajectory).all()


def test_flow_backward_produces_finite_gradients(
    solver_config_factory: object,
    ot_model_config: dict[str, object],
    tiny_training_config: dict[str, object],
    easy_ot_batch: dict[str, torch.Tensor],
) -> None:
    solver = build_solver(
        ot_model_config,
        solver_config_factory("flow"),  # type: ignore[operator]
        tiny_training_config,
    )
    solver.configure_optimizers(total_steps=2)
    terminal, trajectory = solver.integrate_flow(easy_ot_batch["source"], return_trajectory=True)
    loss = (terminal - easy_ot_batch["target"]).pow(2).mean() + solver._path_energy(trajectory)  # type: ignore[attr-defined]
    loss.backward()
    gradients = [parameter.grad for parameter in solver.parameters() if parameter.requires_grad and parameter.grad is not None]
    assert gradients
    assert all(torch.isfinite(gradient).all() for gradient in gradients)
