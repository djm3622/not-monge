from __future__ import annotations

from pathlib import Path

import torch

from src.evaluation.visualization import save_ot_visualizations


def test_save_ot_visualizations_writes_png(tmp_path: Path) -> None:
    points = torch.tensor(
        [
            [-1.0, -1.0, 0.5],
            [0.0, 0.0, 0.5],
            [1.0, 1.0, 0.5],
        ],
        dtype=torch.float32,
    )
    aggregated = {
        "source": points,
        "prediction": points + 0.25,
        "target": points + 0.5,
        "ground_truth_map": points + 0.5,
    }

    image_path = save_ot_visualizations(aggregated, tmp_path, max_items=3)

    assert image_path == tmp_path / "transport_geometry.png"
    assert image_path.exists()
    assert image_path.stat().st_size > 0
    assert (tmp_path / "transport_geometry.pdf").exists()


class _LinearTransport:
    def compute_map(self, x: torch.Tensor) -> torch.Tensor:
        return x + 0.5


def test_save_ot_visualizations_writes_saddle_outputs_for_forward_only_solver(tmp_path: Path) -> None:
    points = torch.tensor(
        [
            [-1.0, -1.0, 0.5],
            [0.0, 0.0, 0.5],
            [1.0, 1.0, 0.5],
            [2.0, 2.0, 0.5],
        ],
        dtype=torch.float32,
    )
    aggregated = {
        "source": points,
        "prediction": points + 0.25,
        "target": points + 0.5,
        "ground_truth_map": points + 0.5,
    }

    save_ot_visualizations(aggregated, tmp_path, max_items=4, solver=_LinearTransport(), saddle_examples=2)

    assert (tmp_path / "saddle_geometry.png").exists()
    assert (tmp_path / "saddle_geometry.pdf").exists()
    assert (tmp_path / "saddle_samples" / "saddle_point_01.png").exists()
    assert (tmp_path / "saddle_samples" / "saddle_point_02.png").exists()
