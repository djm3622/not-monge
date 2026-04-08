from __future__ import annotations

import copy

import pytest
import torch

from src.solvers.registry import build_solver
from src.training.diffusion_task import DiffusionTrainingTask

pytestmark = pytest.mark.integration


def _assert_gradients_present(parameters: list[torch.nn.Parameter]) -> None:
    assert parameters
    grads = [parameter.grad for parameter in parameters if parameter.requires_grad]
    present_grads = [gradient for gradient in grads if gradient is not None]
    assert present_grads
    assert all(torch.isfinite(gradient).all() for gradient in present_grads)
    total_norm = sum(float(gradient.abs().sum()) for gradient in present_grads)
    assert total_norm > 0.0


def _optimizer_parameters(optimizer: torch.optim.Optimizer) -> list[torch.nn.Parameter]:
    return [
        parameter
        for group in optimizer.param_groups
        for parameter in group["params"]
        if parameter.requires_grad
    ]


@pytest.mark.parametrize(
    "solver_name",
    ["minimax", "icnn", "makkuva_icnn_cvx", "makkuva_mlp_ablation", "mm", "mmv2"],
)
def test_multi_optimizer_solvers_receive_gradients(
    solver_name: str,
    solver_config_factory: object,
    ot_model_config: dict[str, object],
    tiny_training_config: dict[str, object],
    easy_ot_batch: dict[str, torch.Tensor],
    disabled_grad_scaler: torch.amp.GradScaler,
    null_autocast: object,
) -> None:
    solver = build_solver(
        ot_model_config,
        solver_config_factory(solver_name),  # type: ignore[operator]
        tiny_training_config,
    )
    solver.configure_optimizers(total_steps=2)
    for parameter in solver.parameters():
        parameter.grad = None
    solver.training_step(
        easy_ot_batch,
        scaler=disabled_grad_scaler,
        autocast_context=null_autocast,
        gradient_clip_norm=1.0,
    )
    for optimizer in solver.optimizers:
        _assert_gradients_present(_optimizer_parameters(optimizer))


def test_w1_gradients_flow_to_critic_then_step_network(
    solver_config_factory: object,
    ot_model_config: dict[str, object],
    tiny_training_config: dict[str, object],
    easy_ot_batch: dict[str, torch.Tensor],
    disabled_grad_scaler: torch.amp.GradScaler,
    null_autocast: object,
) -> None:
    training_config = copy.deepcopy(tiny_training_config)
    solver_config = solver_config_factory("w1")  # type: ignore[operator]
    solver_config["phase1_ratio"] = 0.25
    solver = build_solver(ot_model_config, solver_config, training_config)
    solver.configure_optimizers(total_steps=4)

    for parameter in solver.parameters():
        parameter.grad = None
    solver.training_step(
        easy_ot_batch,
        scaler=disabled_grad_scaler,
        autocast_context=null_autocast,
        gradient_clip_norm=1.0,
    )
    _assert_gradients_present([parameter for parameter in solver.critic.parameters() if parameter.requires_grad])
    step_grads = [parameter.grad for parameter in solver.step_network.parameters() if parameter.requires_grad]
    assert all(gradient is None or float(gradient.abs().sum()) == 0.0 for gradient in step_grads)

    for parameter in solver.parameters():
        parameter.grad = None
    solver.training_step(
        easy_ot_batch,
        scaler=disabled_grad_scaler,
        autocast_context=null_autocast,
        gradient_clip_norm=1.0,
    )
    _assert_gradients_present(
        [parameter for parameter in solver.step_network.parameters() if parameter.requires_grad]
    )


def test_diffusion_gradients_flow_to_model_parameters(
    tiny_diffusion_model_config: dict[str, object],
    tiny_training_config: dict[str, object],
    diffusion_batch: dict[str, torch.Tensor],
    disabled_grad_scaler: torch.amp.GradScaler,
    null_autocast: object,
) -> None:
    task = DiffusionTrainingTask(
        model_config=copy.deepcopy(tiny_diffusion_model_config),
        training_config=copy.deepcopy(tiny_training_config),
    )
    task.configure_optimizers(total_steps=2)
    for parameter in task.model.parameters():
        parameter.grad = None
    task.training_step(
        diffusion_batch,
        scaler=disabled_grad_scaler,
        autocast_context=null_autocast,
        gradient_clip_norm=1.0,
    )
    _assert_gradients_present(
        [parameter for parameter in task.model.parameters() if parameter.requires_grad]
    )
