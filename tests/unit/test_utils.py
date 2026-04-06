from __future__ import annotations

import json
import random
from pathlib import Path

import numpy as np
import pytest
import torch
from torch import nn

from src.evaluation.ot_metrics import saddle_residual
from src.training.losses import (
    ddpm_noise_prediction_loss,
    minimax_potential_objective,
    minimax_transport_objective,
    quadratic_cost,
)
from src.training.schedulers import build_one_cycle_schedulers
from src.training.trainer import mean_metrics, move_to_device
from src.utils.checkpointing import save_checkpoint
from src.utils.device import (
    PrecisionConfig,
    infer_device,
    make_autocast_context,
    make_grad_scaler,
    maybe_compile_module,
)
from src.utils.linalg import (
    gaussian_ot_linear_map,
    matrix_symmetric_eig,
    symmetric_matrix_square_root,
)
from src.utils.logging import ConsoleLogger, build_logger
from src.utils.seed import seed_all

pytestmark = pytest.mark.unit


def test_seed_all_repeats_python_numpy_and_torch_streams() -> None:
    seed_all(42, deterministic=True)
    first = (random.random(), np.random.rand(), torch.rand(3))
    seed_all(42, deterministic=True)
    second = (random.random(), np.random.rand(), torch.rand(3))
    assert first[0] == pytest.approx(second[0])
    assert first[1] == pytest.approx(second[1])
    assert torch.allclose(first[2], second[2])


def test_infer_device_returns_cpu_when_accelerators_unavailable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    if hasattr(torch.backends, "mps") and hasattr(torch.backends.mps, "is_available"):
        monkeypatch.setattr(torch.backends.mps, "is_available", lambda: False)
    assert infer_device("auto").type == "cpu"
    assert infer_device("cpu").type == "cpu"


def test_precision_helpers_return_context_and_disabled_cpu_scaler() -> None:
    precision = PrecisionConfig("fp32")
    with make_autocast_context(torch.device("cpu"), precision):
        value = torch.tensor(1.0) + 1.0
    scaler = make_grad_scaler(torch.device("cpu"), precision)
    assert value.item() == pytest.approx(2.0)
    assert scaler.is_enabled() is False


def test_maybe_compile_module_paths(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = nn.Linear(2, 2)
    assert maybe_compile_module(module, enabled=False) is module

    class Wrapped(nn.Module):
        def __init__(self, inner: nn.Module) -> None:
            super().__init__()
            self.inner = inner

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return self.inner(x)

    monkeypatch.setattr(torch, "compile", lambda mod, dynamic=False, fullgraph=False: Wrapped(mod))
    compiled = maybe_compile_module(module, enabled=True)
    assert isinstance(compiled, Wrapped)


def test_linalg_helpers_are_consistent() -> None:
    matrix = torch.tensor([[2.0, 0.5], [0.5, 1.5]])
    eigenvalues, eigenvectors = matrix_symmetric_eig(matrix)
    assert torch.all(eigenvalues > 0.0)
    reconstructed = eigenvectors @ torch.diag(eigenvalues) @ eigenvectors.transpose(0, 1)
    assert torch.allclose(reconstructed, matrix, atol=1.0e-4, rtol=1.0e-4)

    square_root = symmetric_matrix_square_root(matrix)
    assert torch.allclose(square_root @ square_root, matrix, atol=1.0e-4, rtol=1.0e-4)

    identity_map = gaussian_ot_linear_map(matrix, matrix)
    assert torch.allclose(identity_map, torch.eye(2), atol=1.0e-3, rtol=1.0e-3)


def test_save_checkpoint_writes_loadable_file(tmp_path: Path) -> None:
    path = tmp_path / "checkpoints" / "state.pt"
    save_checkpoint({"step": 3, "value": torch.tensor([1.0, 2.0])}, path)
    loaded = torch.load(path, map_location="cpu")
    assert loaded["step"] == 3
    assert torch.equal(loaded["value"], torch.tensor([1.0, 2.0]))


def test_console_logger_persists_metrics_and_artifacts(tmp_path: Path) -> None:
    logger = ConsoleLogger(output_dir=tmp_path)
    logger.log_metrics({"train/loss": 1.0}, step=2)
    logger.log_artifact("example", {"status": "ok"}, step=2)
    logger.finalize()

    metrics_path = tmp_path / "metrics.jsonl"
    artifact_path = tmp_path / "artifacts" / "00000002_example.json"
    assert metrics_path.exists()
    assert artifact_path.exists()
    payload = json.loads(metrics_path.read_text(encoding="utf-8").strip())
    assert payload["step"] == 2
    assert payload["train/loss"] == pytest.approx(1.0)


def test_build_logger_wandb_uses_fake_module(
    tmp_path: Path,
    fake_wandb_module: object,
) -> None:
    logger = build_logger(
        backend="wandb",
        output_dir=tmp_path,
        run_name="test-run",
        project="tests",
        config={"x": 1},
        wandb_mode="offline",
    )
    logger.log_metrics({"metric": 2.0}, step=1)
    logger.finalize()
    run = fake_wandb_module.runs[0]  # type: ignore[attr-defined]
    assert run.kwargs["mode"] == "offline"
    assert run.logged[0][1]["metric"] == pytest.approx(2.0)
    assert run.finished is True


def test_training_losses_and_scheduler_are_finite() -> None:
    source = torch.randn(4, 2)
    target = torch.randn(4, 2)
    transported = torch.randn(4, 2)
    potential_real = torch.randn(4, 1)
    potential_fake = torch.randn(4, 1)
    assert torch.isfinite(quadratic_cost(source, target)).all()
    assert torch.isfinite(minimax_potential_objective(potential_real, potential_fake))
    assert torch.isfinite(minimax_transport_objective(source, transported, potential_fake))
    assert torch.isfinite(ddpm_noise_prediction_loss(torch.randn(2, 3, 4, 4), torch.randn(2, 3, 4, 4)))

    module = nn.Linear(2, 2)
    optimizer = torch.optim.AdamW(module.parameters(), lr=1.0e-3)
    scheduler = build_one_cycle_schedulers([optimizer], total_steps=3, max_lrs=[1.0e-3])[0]
    optimizer.zero_grad(set_to_none=True)
    module(torch.randn(2, 2)).sum().backward()
    optimizer.step()
    scheduler.step()
    assert scheduler.get_last_lr()[0] > 0.0


def test_saddle_residual_matches_inverse_response_consistency() -> None:
    targets = torch.randn(8, 3)

    def forward_map(x: torch.Tensor) -> torch.Tensor:
        return 2.0 * x

    def exact_inverse(y: torch.Tensor) -> torch.Tensor:
        return 0.5 * y

    def bad_inverse(y: torch.Tensor) -> torch.Tensor:
        return y

    assert saddle_residual(forward_map, exact_inverse, targets) == pytest.approx(0.0)
    assert saddle_residual(forward_map, bad_inverse, targets) > 0.0


def test_move_to_device_and_mean_metrics_work_recursively() -> None:
    batch = {
        "source": torch.randn(2, 2),
        "items": [torch.randn(1), {"target": torch.randn(2)}],
    }
    moved = move_to_device(batch, torch.device("cpu"))
    assert moved["source"].device.type == "cpu"
    assert moved["items"][0].device.type == "cpu"
    averaged = mean_metrics([{"a": 1.0, "b": 2.0}, {"a": 3.0}])
    assert averaged["a"] == pytest.approx(2.0)
    assert averaged["b"] == pytest.approx(1.0)
