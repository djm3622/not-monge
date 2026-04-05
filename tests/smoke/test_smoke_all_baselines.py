from __future__ import annotations

import pytest
import torch

from src.solvers.registry import build_solver, registered_solver_ids

pytestmark = pytest.mark.smoke


@pytest.mark.parametrize("solver_name", registered_solver_ids())
def test_all_baselines_smoke(
    solver_name: str,
    solver_config_factory: object,
    ot_model_config: dict[str, object],
    tiny_training_config: dict[str, object],
    ot_batch: dict[str, torch.Tensor],
    synthetic_bundle: object,
    disabled_grad_scaler: torch.amp.GradScaler,
    null_autocast: object,
) -> None:
    solver = build_solver(
        ot_model_config,
        solver_config_factory(solver_name),  # type: ignore[operator]
        tiny_training_config,
    )
    if solver.supports_training:
        solver.configure_optimizers(total_steps=2)
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
