from __future__ import annotations

import copy
from pathlib import Path

import pytest
import torch

from src.training.diffusion_task import DiffusionTrainingTask
from src.training.trainer import Trainer

pytestmark = pytest.mark.integration


def test_diffusion_pipeline_runs_end_to_end(
    tiny_diffusion_model_config: dict[str, object],
    tiny_training_config: dict[str, object],
    tiny_fake_image_dataset_config: dict[str, object],
    fake_image_bundle: object,
    tmp_output_dir: Path,
) -> None:
    train_loader, val_loader, _ = fake_image_bundle.make_dataloaders()
    output_dir = tmp_output_dir / "diffusion"
    training_config = copy.deepcopy(tiny_training_config)
    task = DiffusionTrainingTask(
        model_config=copy.deepcopy(tiny_diffusion_model_config),
        training_config=training_config,
    )
    trainer = Trainer(
        config=training_config,
        experiment_name="diffusion_test",
        output_dir=output_dir,
        full_config={
            "model": tiny_diffusion_model_config,
            "training": training_config,
            "dataset": tiny_fake_image_dataset_config,
            "experiment": {"name": "diffusion_test", "output_dir": str(output_dir)},
        },
    )
    metrics = trainer.fit(task, train_loader, val_loader)
    assert "val/ddpm_loss" in metrics
    assert torch.isfinite(torch.tensor(metrics["val/ddpm_loss"]))
    assert (output_dir / "checkpoints" / "epoch_0001.pt").exists()

    samples = task.sample(
        num_samples=4,
        device=torch.device("cpu"),
        num_inference_steps=4,
        batch_size=2,
    )
    assert samples.shape == (4, 3, 16, 16)
    assert torch.isfinite(samples).all()
