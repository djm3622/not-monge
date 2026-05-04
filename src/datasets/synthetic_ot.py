"""Synthetic continuous OT benchmark with analytic ground-truth maps."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any, Iterator, Mapping

import torch
from torch import nn
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
class SyntheticSourceSampler:
    """Fixed source distribution sampler shared across benchmark splits."""

    input_dim: int
    distribution: str
    source_scale: float
    mixture_components: int
    component_means: torch.Tensor | None = None
    component_probs: torch.Tensor | None = None
    component_scales: torch.Tensor | None = None

    def sample(self, num_samples: int, generator: torch.Generator) -> torch.Tensor:
        if self.distribution == "gaussian":
            return self.source_scale * torch.randn(num_samples, self.input_dim, generator=generator)
        if self.distribution == "uniform":
            return self.source_scale * (2.0 * torch.rand(num_samples, self.input_dim, generator=generator) - 1.0)
        if self.component_means is None:
            raise ValueError("Mixture source sampler requires fixed component means")
        if self.component_probs is None:
            assignments = torch.randint(self.mixture_components, (num_samples,), generator=generator)
        else:
            assignments = torch.multinomial(self.component_probs, num_samples, replacement=True, generator=generator)
        noise = 0.35 * self.source_scale * torch.randn(num_samples, self.input_dim, generator=generator)
        if self.component_scales is not None:
            noise = noise * self.component_scales[assignments]
        return self.component_means[assignments] + noise


@dataclass
class SyntheticOTBenchmark:
    """Container for the synthetic OT benchmark splits and ground truth."""

    train: TensorDictDataset | None
    train_loader: Any
    val: TensorDictDataset
    test: TensorDictDataset
    ground_truth_potential: nn.Module
    batch_size: int
    num_workers: int
    source_rms: float
    target_rms: float
    potential_scale: float

    def make_dataloaders(self) -> tuple[Any, ...]:
        loader_kwargs = {
            "batch_size": self.batch_size,
            "num_workers": self.num_workers,
            "pin_memory": False,
        }
        val_loader = DataLoader(self.val, shuffle=False, drop_last=False, **loader_kwargs)
        test_loader = DataLoader(self.test, shuffle=False, drop_last=False, **loader_kwargs)
        return self.train_loader, val_loader, test_loader


class SyntheticResampledBatchLoader:
    """Epoch-wise resampled synthetic training loader with fixed validation/test splits."""

    def __init__(
        self,
        *,
        potential: nn.Module,
        source_sampler: SyntheticSourceSampler,
        batch_size: int,
        steps_per_epoch: int,
        seed: int,
        epoch_seed_stride: int,
    ) -> None:
        self.potential = potential
        self.source_sampler = source_sampler
        self.batch_size = int(batch_size)
        self.steps_per_epoch = int(steps_per_epoch)
        self.seed = int(seed)
        self.epoch_seed_stride = int(epoch_seed_stride)
        self._epoch_index = 0

    def __iter__(self) -> Iterator[dict[str, torch.Tensor]]:
        epoch_seed = self.seed + self._epoch_index * self.epoch_seed_stride
        self._epoch_index += 1
        rng = torch.Generator().manual_seed(epoch_seed)
        for _ in range(self.steps_per_epoch):
            source = self.source_sampler.sample(self.batch_size, rng)
            target = batched_potential_gradient(self.potential, source)
            yield {
                "source": source,
                "target": target,
                "ground_truth_map": target,
            }

    def __len__(self) -> int:
        return self.steps_per_epoch


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


def build_source_sampler(config: Mapping[str, Any], rng: torch.Generator) -> SyntheticSourceSampler:
    """Construct a fixed source sampler for all benchmark splits."""
    distribution = str(config.get("source_distribution", "gaussian_mixture"))
    input_dim = int(config["input_dim"])
    source_scale = float(config.get("source_scale", 1.0))
    mixture_components = int(config.get("mixture_components", 4))
    if distribution in {"gaussian", "uniform"}:
        return SyntheticSourceSampler(
            input_dim=input_dim,
            distribution=distribution,
            source_scale=source_scale,
            mixture_components=mixture_components,
        )
    component_means = torch.randn(mixture_components, input_dim, generator=rng)
    component_means = 2.5 * source_scale * component_means / component_means.norm(
        dim=-1,
        keepdim=True,
    ).clamp_min(1e-6)
    component_probs: torch.Tensor | None = None
    mixture_weights = config.get("mixture_weights")
    if mixture_weights is not None:
        component_probs = torch.as_tensor(mixture_weights, dtype=torch.float32)
        if component_probs.numel() != mixture_components:
            raise ValueError("mixture_weights must have length equal to mixture_components")
        component_probs = component_probs / component_probs.sum().clamp_min(1e-12)
    else:
        logits_std = float(config.get("mixture_weight_logits_std", 0.0))
        if logits_std > 0.0:
            logits = logits_std * torch.randn(mixture_components, generator=rng)
            component_probs = torch.softmax(logits, dim=0)

    component_scales: torch.Tensor | None = None
    covariance_mode = str(config.get("mixture_covariance_mode", "isotropic"))
    covariance_log_std = float(config.get("mixture_covariance_log_std", 0.0))
    if covariance_mode not in {"isotropic", "anisotropic_diag"}:
        raise ValueError(f"Unsupported mixture_covariance_mode: {covariance_mode}")
    if covariance_mode == "anisotropic_diag":
        if covariance_log_std > 0.0:
            log_scales = covariance_log_std * torch.randn(mixture_components, input_dim, generator=rng)
            log_scales = log_scales - log_scales.mean(dim=-1, keepdim=True)
            component_scales = torch.exp(log_scales)
        else:
            component_scales = torch.ones(mixture_components, input_dim)
    return SyntheticSourceSampler(
        input_dim=input_dim,
        distribution=distribution,
        source_scale=source_scale,
        mixture_components=mixture_components,
        component_means=component_means,
        component_probs=component_probs,
        component_scales=component_scales,
    )


def batched_potential_gradient(
    potential: nn.Module,
    inputs: torch.Tensor,
    batch_size: int = 2048,
) -> torch.Tensor:
    """Apply the ground-truth gradient map in manageable chunks."""
    gradients: list[torch.Tensor] = []
    for chunk in inputs.split(batch_size):
        chunk = chunk.clone().requires_grad_(True)
        gradients.append(potential.gradient(chunk).detach())
    return torch.cat(gradients, dim=0)


class ScaledPotential(nn.Module):
    """Wrap a scalar potential and scale both its values and gradients."""

    def __init__(self, backbone: nn.Module, scale: float) -> None:
        super().__init__()
        self.backbone = backbone
        self.scale = float(scale)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.scale * self.backbone(x)

    def gradient(self, x: torch.Tensor, create_graph: bool = False) -> torch.Tensor:
        if hasattr(self.backbone, "gradient"):
            return self.scale * self.backbone.gradient(x, create_graph=create_graph)
        x = x.requires_grad_(True)
        values = self.forward(x)
        return torch.autograd.grad(values.sum(), x, create_graph=create_graph)[0]


def _vector_rms(values: torch.Tensor) -> float:
    return float(values.pow(2).sum(dim=-1).mean().sqrt())


def _resolve_potential_scale(
    potential: nn.Module,
    *,
    source_sampler: SyntheticSourceSampler,
    config: Mapping[str, Any],
    seed: int,
) -> tuple[float, float, float]:
    calibration_samples = int(config.get("calibration_samples", 4096))
    if calibration_samples <= 0:
        return 1.0, math.nan, math.nan

    calibration_rng = torch.Generator().manual_seed(seed + 1_000_003)
    source = source_sampler.sample(calibration_samples, calibration_rng)
    gradients = batched_potential_gradient(potential, source)
    source_rms = _vector_rms(source)
    target_rms = _vector_rms(gradients)

    desired_target_rms = config.get("target_rms")
    if desired_target_rms is None:
        multiplier = float(config.get("target_rms_multiplier", 0.0))
        if multiplier <= 0.0:
            return 1.0, source_rms, target_rms
        desired_target_rms = multiplier * source_rms
    desired_target_rms = float(desired_target_rms)
    if target_rms <= 0.0 or not torch.isfinite(torch.tensor(target_rms)):
        return 1.0, source_rms, target_rms
    return desired_target_rms / target_rms, source_rms, target_rms


def make_split(
    potential: nn.Module,
    num_samples: int,
    source_sampler: SyntheticSourceSampler,
    rng: torch.Generator,
) -> TensorDictDataset:
    """Construct one split of the benchmark."""
    source = source_sampler.sample(num_samples, rng)
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
    source_sampler = build_source_sampler(cfg, rng)
    potential = InputConvexNeuralNetwork(
        input_dim=int(cfg["input_dim"]),
        hidden_dims=list(cfg["generator_hidden_dims"]),
        activation="softplus",
        strong_convexity=float(cfg.get("strong_convexity", 0.1)),
    )
    init_std = float(cfg.get("generator_init_std", 0.1))
    with torch.no_grad():
        for parameter in potential.parameters():
            parameter.copy_(
                init_std
                * torch.randn(
                    parameter.shape,
                    generator=rng,
                    device=parameter.device,
                    dtype=parameter.dtype,
                )
            )
    scale_factor, source_rms, unscaled_target_rms = _resolve_potential_scale(
        potential,
        source_sampler=source_sampler,
        config=cfg,
        seed=seed,
    )
    scaled_potential: nn.Module = ScaledPotential(potential, scale_factor)
    scaled_potential.eval()
    for parameter in scaled_potential.parameters():
        parameter.requires_grad_(False)
    target_rms = unscaled_target_rms * scale_factor if torch.isfinite(torch.tensor(unscaled_target_rms)) else math.nan
    batch_size = int(cfg.get("batch_size", 512))
    steps_per_epoch = int(cfg.get("steps_per_epoch", max(int(cfg["n_train"]) // max(batch_size, 1), 1)))
    resample_train = bool(cfg.get("resample_train", True))
    train_dataset = (
        None
        if resample_train
        else make_split(
            potential=scaled_potential,
            num_samples=int(cfg["n_train"]),
            source_sampler=source_sampler,
            rng=rng,
        )
    )
    if resample_train:
        train_loader: Any = SyntheticResampledBatchLoader(
            potential=scaled_potential,
            source_sampler=source_sampler,
            batch_size=batch_size,
            steps_per_epoch=steps_per_epoch,
            seed=seed,
            epoch_seed_stride=int(cfg.get("train_epoch_seed_stride", 1_000_003)),
        )
    else:
        assert train_dataset is not None
        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            num_workers=int(cfg.get("num_workers", 0)),
            pin_memory=False,
            shuffle=True,
            drop_last=True,
        )
    return SyntheticOTBenchmark(
        train=train_dataset,
        train_loader=train_loader,
        val=make_split(
            potential=scaled_potential,
            num_samples=int(cfg["n_val"]),
            source_sampler=source_sampler,
            rng=rng,
        ),
        test=make_split(
            potential=scaled_potential,
            num_samples=int(cfg["n_test"]),
            source_sampler=source_sampler,
            rng=rng,
        ),
        ground_truth_potential=scaled_potential,
        batch_size=batch_size,
        num_workers=int(cfg.get("num_workers", 0)),
        source_rms=source_rms,
        target_rms=target_rms,
        potential_scale=float(scale_factor),
    )
