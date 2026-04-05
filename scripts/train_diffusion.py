"""Train a DDPM on image datasets."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import hydra
from omegaconf import DictConfig, OmegaConf

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.datasets.celeba import build_image_dataset_bundle
from src.training.diffusion_task import DiffusionTrainingTask
from src.training.trainer import Trainer


@hydra.main(version_base=None, config_path="../configs", config_name="diffusion")
def main(cfg: DictConfig) -> None:
    config = OmegaConf.to_container(cfg, resolve=True)
    assert isinstance(config, dict)
    data_bundle = build_image_dataset_bundle(dict(config["dataset"]))
    train_loader, val_loader, _ = data_bundle.make_dataloaders()
    task = DiffusionTrainingTask(config["model"], config["training"])
    trainer = Trainer(
        config=config["training"],
        experiment_name=str(config["experiment"]["name"]),
        output_dir=ROOT / str(config["experiment"]["output_dir"]),
        full_config=config,
    )
    metrics = trainer.fit(task, train_loader, val_loader)
    output_dir = ROOT / str(config["experiment"]["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "val_metrics.json").open("w", encoding="utf-8") as handle:
        json.dump(metrics, handle, indent=2)
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
