"""Two-dimensional Makkuva et al. benchmark datasets."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
from torch.utils.data import DataLoader

from src.datasets.synthetic_ot import TensorDictDataset


def _checkerboard_centers(name: str, scale: float) -> torch.Tensor:
    if name == "checker_board_five":
        return scale * torch.tensor(
            [
                [0.0, 0.0],
                [1.0, 1.0],
                [-1.0, 1.0],
                [-1.0, -1.0],
                [1.0, -1.0],
            ],
            dtype=torch.float32,
        )
    if name == "checker_board_four":
        return scale * torch.tensor(
            [
                [1.0, 0.0],
                [0.0, 1.0],
                [-1.0, 0.0],
                [0.0, -1.0],
            ],
            dtype=torch.float32,
        )
    raise ValueError(f"Unsupported Makkuva distribution '{name}'")


def _sample_checkerboard(
    name: str,
    num_samples: int,
    *,
    scale: float,
    variance: float,
    generator: torch.Generator,
) -> torch.Tensor:
    centers = _checkerboard_centers(name, scale=scale)
    assignments = torch.randint(centers.shape[0], (num_samples,), generator=generator)
    offsets = variance * (2.0 * torch.rand(num_samples, 2, generator=generator) - 1.0)
    return centers[assignments] + offsets


@dataclass(frozen=True)
class AffineStandardizer:
    """Fixed affine transform shared across source and target samples."""

    mean: torch.Tensor
    scale: torch.Tensor

    def transform(self, values: torch.Tensor) -> torch.Tensor:
        return (values - self.mean) / self.scale


class MakkuvaTrainLoader:
    """Infinite-sampler style loader with a fixed epoch length."""

    def __init__(
        self,
        *,
        source_distribution: str,
        target_distribution: str,
        batch_size: int,
        steps_per_epoch: int,
        scale: float,
        variance: float,
        seed: int,
        standardizer: AffineStandardizer,
    ) -> None:
        self.source_distribution = source_distribution
        self.target_distribution = target_distribution
        self.batch_size = int(batch_size)
        self.steps_per_epoch = int(steps_per_epoch)
        self.scale = float(scale)
        self.variance = float(variance)
        self.seed = int(seed)
        self.standardizer = standardizer

    def __len__(self) -> int:
        return self.steps_per_epoch

    def __iter__(self) -> Any:
        generator = torch.Generator().manual_seed(self.seed)
        for _ in range(self.steps_per_epoch):
            source = _sample_checkerboard(
                self.source_distribution,
                self.batch_size,
                scale=self.scale,
                variance=self.variance,
                generator=generator,
            )
            target = _sample_checkerboard(
                self.target_distribution,
                self.batch_size,
                scale=self.scale,
                variance=self.variance,
                generator=generator,
            )
            yield {
                "source": self.standardizer.transform(source),
                "target": self.standardizer.transform(target),
            }


@dataclass
class Makkuva2DBenchmark:
    """Container for train/validation/test loaders for Makkuva 2D OT."""

    train_loader: MakkuvaTrainLoader
    val: TensorDictDataset
    test: TensorDictDataset
    batch_size: int
    num_workers: int
    standardizer: AffineStandardizer

    def make_dataloaders(self) -> tuple[Any, ...]:
        loader_kwargs = {
            "batch_size": self.batch_size,
            "num_workers": self.num_workers,
            "pin_memory": False,
        }
        val_loader = DataLoader(self.val, shuffle=False, drop_last=False, **loader_kwargs)
        test_loader = DataLoader(self.test, shuffle=False, drop_last=False, **loader_kwargs)
        return self.train_loader, val_loader, test_loader


def _estimate_standardizer(
    *,
    source_distribution: str,
    target_distribution: str,
    standardize_samples: int,
    scale: float,
    variance: float,
    seed: int,
) -> AffineStandardizer:
    generator = torch.Generator().manual_seed(seed)
    source = _sample_checkerboard(
        source_distribution,
        standardize_samples,
        scale=scale,
        variance=variance,
        generator=generator,
    )
    target = _sample_checkerboard(
        target_distribution,
        standardize_samples,
        scale=scale,
        variance=variance,
        generator=generator,
    )
    stacked = torch.cat([source, target], dim=0)
    mean = stacked.mean(dim=0)
    scale_tensor = stacked.std(dim=0, unbiased=False).clamp_min(1.0e-6)
    return AffineStandardizer(mean=mean, scale=scale_tensor)


def _build_fixed_split(
    *,
    source_distribution: str,
    target_distribution: str,
    num_samples: int,
    scale: float,
    variance: float,
    seed: int,
    standardizer: AffineStandardizer,
) -> TensorDictDataset:
    generator = torch.Generator().manual_seed(seed)
    source = _sample_checkerboard(
        source_distribution,
        num_samples,
        scale=scale,
        variance=variance,
        generator=generator,
    )
    target = _sample_checkerboard(
        target_distribution,
        num_samples,
        scale=scale,
        variance=variance,
        generator=generator,
    )
    return TensorDictDataset(
        {
            "source": standardizer.transform(source),
            "target": standardizer.transform(target),
        }
    )


def build_makkuva_2d_benchmark(config: dict[str, Any]) -> Makkuva2DBenchmark:
    """Build the two-dimensional Makkuva checkerboard OT benchmark."""
    cfg = dict(config)
    source_distribution = str(cfg.get("source_distribution", "checker_board_five"))
    target_distribution = str(cfg.get("target_distribution", "checker_board_four"))
    scale = float(cfg.get("scale", 1.0))
    variance = float(cfg.get("variance", 0.5))
    seed = int(cfg.get("seed", 1234))
    standardizer = _estimate_standardizer(
        source_distribution=source_distribution,
        target_distribution=target_distribution,
        standardize_samples=int(cfg.get("standardize_samples", 16384)),
        scale=scale,
        variance=variance,
        seed=seed + 7,
    )
    return Makkuva2DBenchmark(
        train_loader=MakkuvaTrainLoader(
            source_distribution=source_distribution,
            target_distribution=target_distribution,
            batch_size=int(cfg.get("batch_size", 1024)),
            steps_per_epoch=int(cfg.get("steps_per_epoch", 100)),
            scale=scale,
            variance=variance,
            seed=seed + 11,
            standardizer=standardizer,
        ),
        val=_build_fixed_split(
            source_distribution=source_distribution,
            target_distribution=target_distribution,
            num_samples=int(cfg.get("n_val", 2048)),
            scale=scale,
            variance=variance,
            seed=seed + 22,
            standardizer=standardizer,
        ),
        test=_build_fixed_split(
            source_distribution=source_distribution,
            target_distribution=target_distribution,
            num_samples=int(cfg.get("n_test", 2048)),
            scale=scale,
            variance=variance,
            seed=seed + 33,
            standardizer=standardizer,
        ),
        batch_size=int(cfg.get("batch_size", 1024)),
        num_workers=int(cfg.get("num_workers", 0)),
        standardizer=standardizer,
    )
