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
    disabled_grad_scaler: torch.amp.GradScaler,
    null_autocast: object,
) -> None:
    solver = build_solver(
        ot_model_config,
        solver_config_factory("otp"),  # type: ignore[operator]
        tiny_training_config,
    )
    solver.configure_optimizers(total_steps=8)
    potential = solver.compute_potential(ot_batch["target"])
    assert potential is not None
    assert torch.isfinite(potential).all()
    expected = solver.quadratic_scale * ot_batch["target"].pow(2).sum(dim=-1, keepdim=True) - solver.potential_backbone(
        ot_batch["target"]
    )
    assert torch.allclose(potential, expected)
    initial_noise = solver.current_noise_level()
    solver.training_step(
        ot_batch,
        scaler=disabled_grad_scaler,
        autocast_context=null_autocast,
        gradient_clip_norm=None,
    )
    assert solver.current_noise_level() <= initial_noise + 1.0e-8


def test_otp_backward_produces_finite_gradients(
    solver_config_factory: object,
    ot_model_config: dict[str, object],
    tiny_training_config: dict[str, object],
    easy_ot_batch: dict[str, torch.Tensor],
    disabled_grad_scaler: torch.amp.GradScaler,
    null_autocast: object,
) -> None:
    solver = build_solver(
        ot_model_config,
        solver_config_factory("otp"),  # type: ignore[operator]
        tiny_training_config,
    )
    solver.configure_optimizers(total_steps=2)
    solver.training_step(
        easy_ot_batch,
        scaler=disabled_grad_scaler,
        autocast_context=null_autocast,
        gradient_clip_norm=None,
    )
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
