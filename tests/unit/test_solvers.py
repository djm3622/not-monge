from __future__ import annotations

import contextlib
import copy

import pytest
import torch

from src.solvers.registry import build_solver, registered_solver_ids

pytestmark = pytest.mark.unit

EXPECTED_SOLVERS = [
    "minimax",
    "icnn",
    "makkuva_icnn_cvx",
    "makkuva_mlp_ablation",
    "tw2",
    "mmv2",
    "mm",
    "mm_b",
    "qc",
    "sinkhorn",
    "gaussian",
    "entropic",
    "otp",
    "flow",
    "w1",
]


def test_registry_contains_all_required_solver_ids() -> None:
    registered = set(registered_solver_ids())
    assert registered.issuperset(EXPECTED_SOLVERS)


@pytest.mark.parametrize("solver_name", EXPECTED_SOLVERS)
def test_every_solver_builds_and_respects_interface(
    solver_name: str,
    solver_config_factory: object,
    ot_model_config: dict[str, object],
    tiny_training_config: dict[str, object],
    ot_batch: dict[str, torch.Tensor],
    disabled_grad_scaler: torch.amp.GradScaler,
    null_autocast: object,
    synthetic_bundle: object,
) -> None:
    solver = build_solver(
        ot_model_config,
        solver_config_factory(solver_name),  # type: ignore[operator]
        tiny_training_config,
    )
    if solver.supports_training:
        solver.configure_optimizers(total_steps=2)
        mapped = solver.compute_map(ot_batch["source"])
        assert mapped.shape == ot_batch["source"].shape
        assert torch.isfinite(mapped).all()
        potential = solver.compute_potential(ot_batch["source"])
        assert potential is None or potential.shape[0] == ot_batch["source"].shape[0]
        metrics = solver.training_step(
            ot_batch,
            scaler=disabled_grad_scaler,
            autocast_context=null_autocast,
            gradient_clip_norm=1.0,
        )
        assert metrics
        assert all(torch.isfinite(torch.tensor(value)) for value in metrics.values())
    else:
        train_loader, _, _ = synthetic_bundle.make_dataloaders()
        solver.fit_reference(train_loader)
        mapped = solver.compute_map(ot_batch["source"])
        assert mapped.shape == ot_batch["source"].shape
        assert torch.isfinite(mapped).all()
        assert solver.solver_group == "reference"


@pytest.mark.parametrize("solver_name", ["sinkhorn", "gaussian"])
def test_reference_solver_state_dict_round_trip(
    solver_name: str,
    solver_config_factory: object,
    ot_model_config: dict[str, object],
    tiny_training_config: dict[str, object],
    synthetic_bundle: object,
    ot_batch: dict[str, torch.Tensor],
) -> None:
    solver = build_solver(
        ot_model_config,
        solver_config_factory(solver_name),  # type: ignore[operator]
        tiny_training_config,
    )
    train_loader, _, _ = synthetic_bundle.make_dataloaders()
    solver.fit_reference(train_loader)
    baseline = solver.compute_map(ot_batch["source"])
    state = copy.deepcopy(solver.state_dict())
    restored = build_solver(
        ot_model_config,
        solver_config_factory(solver_name),  # type: ignore[operator]
        tiny_training_config,
    )
    restored.load_state_dict(state)
    recovered = restored.compute_map(ot_batch["source"])
    assert torch.allclose(baseline, recovered, atol=1.0e-6, rtol=1.0e-5)


def test_gaussian_reference_is_close_to_identity(
    solver_config_factory: object,
    ot_model_config: dict[str, object],
    tiny_training_config: dict[str, object],
) -> None:
    solver = build_solver(
        ot_model_config,
        solver_config_factory("gaussian"),  # type: ignore[operator]
        tiny_training_config,
    )
    source = torch.randn(32, 2)
    loader = [{"source": source, "target": source.clone()}]
    solver.fit_reference(loader)
    mapped = solver.compute_map(source)
    assert torch.allclose(mapped, source, atol=2.5e-1, rtol=1.0e-4)


def test_sinkhorn_reference_is_near_identity_on_matching_support(
    solver_config_factory: object,
    ot_model_config: dict[str, object],
    tiny_training_config: dict[str, object],
) -> None:
    solver = build_solver(
        ot_model_config,
        solver_config_factory("sinkhorn"),  # type: ignore[operator]
        tiny_training_config,
    )
    support = torch.randn(16, 2)
    solver.fit_reference([{"source": support, "target": support.clone()}])
    mapped = solver.compute_map(support)
    assert mapped.shape == support.shape
    assert torch.mean((mapped - support).pow(2)).sqrt() < 5.0e-1


def test_makkuva_convex_solver_projects_f_and_reports_penalty(
    solver_config_factory: object,
    ot_model_config: dict[str, object],
    tiny_training_config: dict[str, object],
    ot_batch: dict[str, torch.Tensor],
    disabled_grad_scaler: torch.amp.GradScaler,
    null_autocast: object,
) -> None:
    solver = build_solver(
        ot_model_config,
        solver_config_factory("makkuva_icnn_cvx"),  # type: ignore[operator]
        tiny_training_config,
    )
    solver.configure_optimizers(total_steps=2)
    metrics = solver.training_step(
        ot_batch,
        scaler=disabled_grad_scaler,
        autocast_context=null_autocast,
        gradient_clip_norm=1.0,
    )
    assert metrics["train/g_penalty"] >= 0.0
    assert all(torch.all(parameter >= 0.0) for parameter in solver.f_potential.positive_parameters())


def test_makkuva_mlp_ablation_skips_convex_penalty(
    solver_config_factory: object,
    ot_model_config: dict[str, object],
    tiny_training_config: dict[str, object],
    ot_batch: dict[str, torch.Tensor],
    disabled_grad_scaler: torch.amp.GradScaler,
    null_autocast: object,
) -> None:
    solver = build_solver(
        ot_model_config,
        solver_config_factory("makkuva_mlp_ablation"),  # type: ignore[operator]
        tiny_training_config,
    )
    solver.configure_optimizers(total_steps=2)
    metrics = solver.training_step(
        ot_batch,
        scaler=disabled_grad_scaler,
        autocast_context=null_autocast,
        gradient_clip_norm=1.0,
    )
    assert metrics["train/g_penalty"] == pytest.approx(0.0, abs=1.0e-8)
