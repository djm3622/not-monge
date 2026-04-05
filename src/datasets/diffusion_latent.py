"""Latent dataset utilities for diffusion-timestep OT experiments."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import torch
from diffusers import DDPMScheduler
from torch import nn
from torch.utils.data import DataLoader, Dataset

from src.datasets.celeba import build_image_dataset_bundle

try:
    import timm
except ImportError:  # pragma: no cover - optional dependency resolved at install time
    timm = None


class TimmLatentEncoder(nn.Module):
    """Feature encoder used to move images into a compact latent space."""

    def __init__(self, model_name: str, latent_dim: int, pretrained: bool = False) -> None:
        super().__init__()
        if timm is None:
            raise ImportError("timm is required for latent extraction experiments")
        self.backbone = timm.create_model(model_name, pretrained=pretrained, num_classes=0)
        self.projection = nn.Linear(int(self.backbone.num_features), latent_dim)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        features = self.backbone(images)
        return self.projection(features)


class DiffusionLatentDataset(Dataset[dict[str, torch.Tensor]]):
    """Dataset of latent pairs across diffusion timesteps."""

    def __init__(self, records: dict[str, torch.Tensor]) -> None:
        self.records = records
        self.length = next(iter(records.values())).shape[0]

    def __len__(self) -> int:
        return self.length

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        return {key: value[index] for key, value in self.records.items()}


@dataclass
class DiffusionLatentBundle:
    """Split bundle for latent OT experiments."""

    train: DiffusionLatentDataset
    val: DiffusionLatentDataset
    test: DiffusionLatentDataset
    batch_size: int
    num_workers: int
    encoder: nn.Module
    scheduler: DDPMScheduler

    def make_dataloaders(self) -> tuple[DataLoader[dict[str, torch.Tensor]], ...]:
        loader_kwargs = {
            "batch_size": self.batch_size,
            "num_workers": self.num_workers,
            "pin_memory": False,
        }
        train_loader = DataLoader(self.train, shuffle=True, drop_last=True, **loader_kwargs)
        val_loader = DataLoader(self.val, shuffle=False, drop_last=False, **loader_kwargs)
        test_loader = DataLoader(self.test, shuffle=False, drop_last=False, **loader_kwargs)
        return train_loader, val_loader, test_loader


def _materialize_split(
    loader: DataLoader[dict[str, torch.Tensor]],
    encoder: nn.Module,
    scheduler: DDPMScheduler,
    timesteps: list[int],
    save_path: Path | None = None,
) -> DiffusionLatentDataset:
    """Encode images and save noisy latent snapshots for adjacent timestep OT."""
    latent_start: list[torch.Tensor] = []
    latent_end: list[torch.Tensor] = []
    clean_latents: list[torch.Tensor] = []
    timestep_start: list[torch.Tensor] = []
    timestep_end: list[torch.Tensor] = []

    encoder.eval()
    with torch.no_grad():
        for batch in loader:
            images = batch["image"]
            z0 = encoder(images)
            for start_t, end_t in zip(timesteps[:-1], timesteps[1:]):
                noise = torch.randn_like(z0)
                start = scheduler.add_noise(
                    z0,
                    noise,
                    torch.full((z0.shape[0],), start_t, dtype=torch.long),
                )
                end = scheduler.add_noise(
                    z0,
                    noise,
                    torch.full((z0.shape[0],), end_t, dtype=torch.long),
                )
                latent_start.append(start.cpu())
                latent_end.append(end.cpu())
                clean_latents.append(z0.cpu())
                timestep_start.append(torch.full((z0.shape[0],), start_t, dtype=torch.long))
                timestep_end.append(torch.full((z0.shape[0],), end_t, dtype=torch.long))

    records = {
        "source": torch.cat(latent_start, dim=0),
        "target": torch.cat(latent_end, dim=0),
        "clean_latent": torch.cat(clean_latents, dim=0),
        "timestep_start": torch.cat(timestep_start, dim=0),
        "timestep_end": torch.cat(timestep_end, dim=0),
        "ground_truth_map": torch.cat(latent_end, dim=0),
    }
    if save_path is not None:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(records, save_path)
    return DiffusionLatentDataset(records)


def build_diffusion_latent_bundle(config: dict[str, Any]) -> DiffusionLatentBundle:
    """Create cached latent timestep pairs from a source image dataset."""
    image_cfg = {
        "name": config["source_dataset"],
        "root": config.get("root", "data"),
        "image_size": config.get("image_size", 32),
        "download": config.get("download", True),
        "batch_size": config["batch_size"],
        "num_workers": config.get("num_workers", 0),
        "train_size": config.get("train_size", 128),
        "val_size": config.get("val_size", 32),
        "test_size": config.get("test_size", 32),
    }
    image_bundle = build_image_dataset_bundle(image_cfg)
    loader_kwargs = {
        "batch_size": int(config["batch_size"]),
        "num_workers": int(config.get("num_workers", 0)),
        "pin_memory": False,
        "shuffle": False,
        "drop_last": False,
    }
    train_loader = DataLoader(image_bundle.train, **loader_kwargs)
    val_loader = DataLoader(image_bundle.val, **loader_kwargs)
    test_loader = DataLoader(image_bundle.test, **loader_kwargs)
    encoder = TimmLatentEncoder(
        model_name=str(config.get("encoder_name", "resnet18")),
        latent_dim=int(config["latent_dim"]),
        pretrained=bool(config.get("pretrained_encoder", False)),
    )
    scheduler = DDPMScheduler(num_train_timesteps=int(config.get("num_train_timesteps", 1000)))
    timesteps = list(int(timestep) for timestep in config["timesteps"])
    cache_dir = Path(str(config.get("cache_dir", "artifacts/latents")))
    save_intermediates = bool(config.get("save_intermediates", True))
    return DiffusionLatentBundle(
        train=_materialize_split(
            train_loader,
            encoder,
            scheduler,
            timesteps,
            cache_dir / "train.pt" if save_intermediates else None,
        ),
        val=_materialize_split(
            val_loader,
            encoder,
            scheduler,
            timesteps,
            cache_dir / "val.pt" if save_intermediates else None,
        ),
        test=_materialize_split(
            test_loader,
            encoder,
            scheduler,
            timesteps,
            cache_dir / "test.pt" if save_intermediates else None,
        ),
        batch_size=int(config["batch_size"]),
        num_workers=int(config.get("num_workers", 0)),
        encoder=encoder,
        scheduler=scheduler,
    )
