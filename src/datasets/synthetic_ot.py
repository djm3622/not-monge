"""Synthetic continuous OT benchmark with analytic ground-truth maps."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

import torch
from torch.utils.data import DataLoader, Dataset

from src.models.potential import InputConvexNeuralNetwork


class TensorDictDataset(Dataset[dict[str, torch.Tensor]]):
    """Tensor-backed dataset returning sample dictionaries."""

    def __init__(self, tensors: Mapping[str, torch.Tensor]) -> None:
        self.tensors = {key: value.detach().clone() for key, value in tensors.items()}
        lengths = {value.shape[0] for value in self.tensors.values()}
        if len(lengths) != 1:
            raise ValueError("All tensors must have the same number of rows")
        self.length = lengths.pop()

    def __len__(self) -> int:
        return self.length

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        return {key: value[index] for key, value in self.tensors.items()}


@dataclass
class SyntheticOTBenchmark:
    """Container for the synthetic OT benchmark splits and ground truth."""

    train: TensorDictDataset
    val: TensorDictDataset
    test: TensorDictDataset
    ground_truth_potential: InputConvexNeuralNetwork
    batch_size: int
    num_workers: int

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


def sample_source_distribution(
    num_samples: int,
    input_dim: int,
    distribution: str,
    source_scale: float,
    mixture_components: int,
    rng: torch.Generator,
) -> torch.Tensor:
    """Sample a source distribution used by the benchmark."""
    if distribution == "gaussian":
        return source_scale * torch.randn(num_samples, input_dim, generator=rng)
    if distribution == "uniform":
        return source_scale * (2.0 * torch.rand(num_samples, input_dim, generator=rng) - 1.0)

    component_means = torch.randn(mixture_components, input_dim, generator=rng)
    component_means = 2.5 * source_scale * component_means / component_means.norm(
        dim=-1,
        keepdim=True,
    ).clamp_min(1e-6)
    assignments = torch.randint(mixture_components, (num_samples,), generator=rng)
    noise = 0.35 * source_scale * torch.randn(num_samples, input_dim, generator=rng)
    return component_means[assignments] + noise


def batched_potential_gradient(
    potential: InputConvexNeuralNetwork,
    inputs: torch.Tensor,
    batch_size: int = 2048,
) -> torch.Tensor:
    """Apply the ground-truth gradient map in manageable chunks."""
    gradients: list[torch.Tensor] = []
    for chunk in inputs.split(batch_size):
        chunk = chunk.clone().requires_grad_(True)
        gradients.append(potential.gradient(chunk).detach())
    return torch.cat(gradients, dim=0)


def make_split(
    potential: InputConvexNeuralNetwork,
    num_samples: int,
    input_dim: int,
    distribution: str,
    source_scale: float,
    mixture_components: int,
    rng: torch.Generator,
) -> TensorDictDataset:
    """Construct one split of the benchmark."""
    source = sample_source_distribution(
        num_samples=num_samples,
        input_dim=input_dim,
        distribution=distribution,
        source_scale=source_scale,
        mixture_components=mixture_components,
        rng=rng,
    )
    target = batched_potential_gradient(potential, source)
    return TensorDictDataset(
        {
            "source": source,
            "target": target,
            "ground_truth_map": target,
        }
    )


def build_synthetic_ot_benchmark(config: Mapping[str, Any]) -> SyntheticOTBenchmark:
    """Create the ICNN-based synthetic benchmark described in Korotin et al."""
    cfg = dict(config)
    seed = int(cfg.get("seed", 1234))
    rng = torch.Generator().manual_seed(seed)
    potential = InputConvexNeuralNetwork(
        input_dim=int(cfg["input_dim"]),
        hidden_dims=list(cfg["generator_hidden_dims"]),
        activation="softplus",
        strong_convexity=float(cfg.get("strong_convexity", 0.1)),
    )
    with torch.no_grad():
        for parameter in potential.parameters():
            parameter.copy_(
                0.75
                * torch.randn(
                    parameter.shape,
                    generator=rng,
                    device=parameter.device,
                    dtype=parameter.dtype,
                )
            )
    potential.eval()
    for parameter in potential.parameters():
        parameter.requires_grad_(False)
    return SyntheticOTBenchmark(
        train=make_split(
            potential=potential,
            num_samples=int(cfg["n_train"]),
            input_dim=int(cfg["input_dim"]),
            distribution=str(cfg.get("source_distribution", "gaussian_mixture")),
            source_scale=float(cfg.get("source_scale", 1.0)),
            mixture_components=int(cfg.get("mixture_components", 4)),
            rng=rng,
        ),
        val=make_split(
            potential=potential,
            num_samples=int(cfg["n_val"]),
            input_dim=int(cfg["input_dim"]),
            distribution=str(cfg.get("source_distribution", "gaussian_mixture")),
            source_scale=float(cfg.get("source_scale", 1.0)),
            mixture_components=int(cfg.get("mixture_components", 4)),
            rng=rng,
        ),
        test=make_split(
            potential=potential,
            num_samples=int(cfg["n_test"]),
            input_dim=int(cfg["input_dim"]),
            distribution=str(cfg.get("source_distribution", "gaussian_mixture")),
            source_scale=float(cfg.get("source_scale", 1.0)),
            mixture_components=int(cfg.get("mixture_components", 4)),
            rng=rng,
        ),
        ground_truth_potential=potential,
        batch_size=int(cfg.get("batch_size", 512)),
        num_workers=int(cfg.get("num_workers", 0)),
    )
