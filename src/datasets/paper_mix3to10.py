"""Paper-faithful high-dimensional Gaussian-mixture OT benchmark."""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader, Dataset

from src.models.potential import DenseInputConvexNeuralNetwork, build_potential
from src.utils.device import infer_device


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


class WeightedSumPotential(nn.Module):
    """Weighted sum of scalar potentials."""

    def __init__(self, potentials: Sequence[nn.Module], weights: Sequence[float]) -> None:
        super().__init__()
        if len(potentials) != len(weights):
            raise ValueError("Potentials and weights must have the same length")
        self.potentials = nn.ModuleList(potentials)
        self.register_buffer("weights", torch.tensor(weights, dtype=torch.float32))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        outputs = []
        for potential in self.potentials:
            outputs.append(potential(x))
        stacked = torch.stack(outputs, dim=0)
        return (self.weights.view(-1, 1, 1) * stacked).sum(dim=0)

    def gradient(self, x: torch.Tensor, create_graph: bool = False) -> torch.Tensor:
        x = x.requires_grad_(True)
        value = self.forward(x)
        return torch.autograd.grad(value.sum(), x, create_graph=create_graph)[0]


class ShiftScalePotential(nn.Module):
    """Linear-adjusted potential used by the benchmark standardization step."""

    def __init__(self, base: nn.Module, shift: torch.Tensor, scale: float) -> None:
        super().__init__()
        self.base = base
        self.register_buffer("shift", shift.detach().clone().float())
        self.scale = float(scale)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.scale * (self.base(x) + (x * self.shift).sum(dim=-1, keepdim=True))

    def gradient(self, x: torch.Tensor, create_graph: bool = False) -> torch.Tensor:
        base_gradient = _potential_gradient(self.base, x, create_graph=create_graph)
        return self.scale * (base_gradient + self.shift.view(1, -1))


class PotentialPushforwardSampler:
    """Samples from the pushforward of a base sampler under a gradient map."""

    def __init__(self, base_sampler: "GaussianMixtureSampler", potential: nn.Module) -> None:
        self.base_sampler = base_sampler
        self.potential = potential
        self.dim = base_sampler.dim

    def sample(self, num_samples: int, generator: torch.Generator) -> torch.Tensor:
        source = self.base_sampler.sample(num_samples, generator)
        return self.potential.gradient(source.clone(), create_graph=False).detach()


class SamplerBatchLoader:
    """Simple batch loader backed by sampler objects."""

    def __init__(
        self,
        source_sampler: "GaussianMixtureSampler",
        target_sampler: PotentialPushforwardSampler,
        ground_truth_potential: nn.Module,
        batch_size: int,
        steps_per_epoch: int,
        seed: int,
    ) -> None:
        self.source_sampler = source_sampler
        self.target_sampler = target_sampler
        self.ground_truth_potential = ground_truth_potential
        self.batch_size = int(batch_size)
        self.steps_per_epoch = int(steps_per_epoch)
        self.seed = int(seed)

    def __iter__(self) -> Iterable[dict[str, torch.Tensor]]:
        generator = torch.Generator().manual_seed(self.seed)
        for _ in range(self.steps_per_epoch):
            source = self.source_sampler.sample(self.batch_size, generator)
            target = self.target_sampler.sample(self.batch_size, generator)
            ground_truth_map = self.ground_truth_potential.gradient(source.clone(), create_graph=False).detach()
            yield {
                "source": source,
                "target": target,
                "ground_truth_map": ground_truth_map,
            }

    def __len__(self) -> int:
        return self.steps_per_epoch


@dataclass
class GaussianMixtureSampler:
    """Uniform Gaussian mixture with full covariance matrices."""

    means: torch.Tensor
    covariances: torch.Tensor

    @property
    def dim(self) -> int:
        return int(self.means.shape[1])

    def sample(self, num_samples: int, generator: torch.Generator) -> torch.Tensor:
        num_components, input_dim = self.means.shape
        assignments = torch.randint(num_components, (num_samples,), generator=generator)
        samples = torch.empty(num_samples, input_dim, dtype=self.means.dtype)
        eye = torch.eye(input_dim, dtype=self.means.dtype)
        for component in range(num_components):
            mask = assignments == component
            if not bool(mask.any()):
                continue
            count = int(mask.sum().item())
            chol = torch.linalg.cholesky(self.covariances[component] + 1.0e-6 * eye)
            noise = torch.randn(count, input_dim, generator=generator, dtype=self.means.dtype)
            samples[mask] = self.means[component] + noise @ chol.transpose(0, 1)
        return samples


@dataclass
class PaperBenchmarkBundle:
    """Container for the benchmark samplers, splits, and ground-truth potential."""

    train_loader: SamplerBatchLoader
    val: TensorDictDataset
    test: TensorDictDataset
    ground_truth_potential: nn.Module
    batch_size: int
    num_workers: int

    def make_dataloaders(self) -> tuple[Any, ...]:
        kwargs = {
            "batch_size": self.batch_size,
            "num_workers": self.num_workers,
            "pin_memory": False,
        }
        val_loader = DataLoader(self.val, shuffle=False, drop_last=False, **kwargs)
        test_loader = DataLoader(self.test, shuffle=False, drop_last=False, **kwargs)
        return self.train_loader, val_loader, test_loader


def _potential_gradient(module: nn.Module, x: torch.Tensor, create_graph: bool) -> torch.Tensor:
    with torch.enable_grad():
        if hasattr(module, "gradient"):
            return module.gradient(x, create_graph=create_graph)  # type: ignore[return-value]
        x = x.requires_grad_(True)
        value = module(x)
        return torch.autograd.grad(value.sum(), x, create_graph=create_graph)[0]


def _convexify_if_available(module: nn.Module) -> None:
    if hasattr(module, "convexify"):
        module.convexify()  # type: ignore[misc]


def _freeze(module: nn.Module) -> None:
    module.eval()
    for parameter in module.parameters():
        parameter.requires_grad_(False)


def _random_gaussian_mixture(
    input_dim: int,
    num_components: int,
    seed: int,
    grid_spacing: float = 1.0,
    covariance_scale: float = 0.4,
) -> GaussianMixtureSampler:
    generator = torch.Generator().manual_seed(seed)
    grid = torch.linspace(
        -0.5 * grid_spacing * num_components,
        0.5 * grid_spacing * num_components,
        steps=num_components,
        dtype=torch.float32,
    )
    means = torch.empty(num_components, input_dim, dtype=torch.float32)
    for dimension in range(input_dim):
        means[:, dimension] = grid[torch.randperm(num_components, generator=generator)]

    covariances = []
    for _ in range(num_components):
        rows = torch.randn(input_dim, input_dim, generator=generator, dtype=torch.float32)
        rows = rows / rows.norm(dim=1, keepdim=True).clamp_min(1.0e-6)
        covariances.append((covariance_scale**2) * (rows @ rows.transpose(0, 1)))
    covariance_tensor = torch.stack(covariances)

    normalizer = 1.0 / math.sqrt(float(means.pow(2).sum(dim=1).mean() / input_dim + covariance_scale**2))
    return GaussianMixtureSampler(
        means=normalizer * means,
        covariances=(normalizer**2) * covariance_tensor,
    )


def _build_dense_icnn(config: Mapping[str, Any], input_dim: int, constrained: bool) -> DenseInputConvexNeuralNetwork:
    module = build_potential(
        {
            "kind": "denseicnn" if constrained else "denseicnn_u",
            "input_dim": input_dim,
            "hidden_dims": list(config.get("benchmark_hidden_dims", [max(2 * input_dim, 64), max(2 * input_dim, 64), max(input_dim, 32)])),
            "rank": int(config.get("benchmark_rank", 1)),
            "activation": str(config.get("benchmark_activation", "celu")),
            "dropout": float(config.get("benchmark_dropout", 0.0)),
            "strong_convexity": float(config.get("benchmark_strong_convexity", 1.0e-4)),
            "identity_quadratic": 0.0,
            "weights_init_std": float(config.get("benchmark_weights_init_std", 0.1)),
        }
    )
    assert isinstance(module, DenseInputConvexNeuralNetwork)
    _convexify_if_available(module)
    return module


def _identity_pretrain(
    module: nn.Module,
    input_dim: int,
    steps: int,
    batch_size: int,
    learning_rate: float,
    blow: float,
    tol: float,
    seed: int,
    device: torch.device,
) -> None:
    if steps <= 0:
        return
    optimizer = torch.optim.Adam(module.parameters(), lr=learning_rate, weight_decay=1.0e-10)
    generator = torch.Generator().manual_seed(seed)
    module.train(True)
    for _ in range(steps):
        batch = blow * torch.randn(batch_size, input_dim, generator=generator, dtype=torch.float32, device=device)
        batch.requires_grad_(True)
        prediction = _potential_gradient(module, batch, create_graph=True)
        loss = F.mse_loss(prediction, batch.detach())
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        _convexify_if_available(module)
        if float(loss.detach()) < tol:
            break


def _standardize_potential(
    potential: nn.Module,
    source_sampler: GaussianMixtureSampler,
    estimate_size: int,
    seed: int,
) -> ShiftScalePotential:
    generator = torch.Generator().manual_seed(seed)
    source = source_sampler.sample(estimate_size, generator)
    pushed = potential.gradient(source.clone(), create_graph=False).detach()
    mean = pushed.mean(dim=0)
    mean_var = (pushed - mean).var(dim=0, unbiased=False).mean().clamp_min(1.0e-8)
    scale = float(mean_var.rsqrt())
    return ShiftScalePotential(potential, shift=-mean, scale=scale)


def _train_tw2_pair(
    source_sampler: GaussianMixtureSampler,
    target_sampler: GaussianMixtureSampler,
    config: Mapping[str, Any],
    cache_path: Path,
    seed_offset: int,
) -> dict[str, Any]:
    input_dim = source_sampler.dim
    device = infer_device(str(config.get("benchmark_device", "auto")))
    batch_size = int(config.get("benchmark_batch_size", 1024))
    train_steps = int(config.get("benchmark_fit_steps", 50000))
    learning_rate = float(config.get("benchmark_fit_lr", 1.0e-3))
    cycle_weight = float(config.get("benchmark_cycle_weight", input_dim))
    pretrain_steps = int(config.get("benchmark_pretrain_steps", 5000))
    pretrain_batch_size = int(config.get("benchmark_pretrain_batch_size", batch_size))
    pretrain_lr = float(config.get("benchmark_pretrain_lr", 1.0e-3))
    pretrain_blow = float(config.get("benchmark_pretrain_blow", 3.0))
    pretrain_tol = float(config.get("benchmark_pretrain_tol", 1.0e-3))

    forward = _build_dense_icnn(config, input_dim=input_dim, constrained=True).to(device)
    inverse = _build_dense_icnn(config, input_dim=input_dim, constrained=True).to(device)
    _identity_pretrain(
        forward,
        input_dim=input_dim,
        steps=pretrain_steps,
        batch_size=pretrain_batch_size,
        learning_rate=pretrain_lr,
        blow=pretrain_blow,
        tol=pretrain_tol,
        seed=seed_offset + 11,
        device=device,
    )
    inverse.load_state_dict(forward.state_dict())

    optimizer_forward = torch.optim.Adam(forward.parameters(), lr=learning_rate)
    optimizer_inverse = torch.optim.Adam(inverse.parameters(), lr=learning_rate)
    source_generator = torch.Generator().manual_seed(seed_offset + 101)
    target_generator = torch.Generator().manual_seed(seed_offset + 202)

    forward.train(True)
    inverse.train(True)
    for _ in range(train_steps):
        source = source_sampler.sample(batch_size, source_generator).to(device)
        target = target_sampler.sample(batch_size, target_generator).to(device)
        source.requires_grad_(True)
        target.requires_grad_(True)

        inverse_target = _potential_gradient(inverse, target, create_graph=True).detach()
        transport_loss = (forward(source) - forward(inverse_target)).mean()
        cycle_loss = F.mse_loss(
            _potential_gradient(forward, _potential_gradient(inverse, target, create_graph=True), create_graph=True),
            target.detach(),
        )
        cycle_loss = cycle_loss + F.mse_loss(
            _potential_gradient(inverse, _potential_gradient(forward, source, create_graph=True), create_graph=True),
            source.detach(),
        )
        loss = transport_loss + cycle_weight * cycle_loss

        optimizer_forward.zero_grad(set_to_none=True)
        optimizer_inverse.zero_grad(set_to_none=True)
        loss.backward()
        optimizer_forward.step()
        optimizer_inverse.step()
        _convexify_if_available(forward)
        _convexify_if_available(inverse)

    forward = forward.to("cpu")
    inverse = inverse.to("cpu")
    _convexify_if_available(forward)
    _convexify_if_available(inverse)
    torch.save(
        {
            "forward_state": forward.state_dict(),
            "inverse_state": inverse.state_dict(),
            "config": dict(config),
        },
        cache_path,
    )
    return {
        "forward_state": forward.state_dict(),
        "inverse_state": inverse.state_dict(),
    }


def _potential_cache_path(config: Mapping[str, Any]) -> Path:
    input_dim = int(config["input_dim"])
    version = str(config.get("cache_version", "paper_ref"))
    root = Path(str(config.get("cache_dir", "artifacts/paper_benchmarks")))
    return root / f"mix3to10_d{input_dim}_{version}_potentials.pt"


def _split_cache_path(config: Mapping[str, Any]) -> Path:
    input_dim = int(config["input_dim"])
    version = str(config.get("cache_version", "paper_ref"))
    root = Path(str(config.get("cache_dir", "artifacts/paper_benchmarks")))
    return root / f"mix3to10_d{input_dim}_{version}_splits.pt"


def _load_ground_truth_potential(config: Mapping[str, Any]) -> tuple[GaussianMixtureSampler, nn.Module]:
    cfg = dict(config)
    input_dim = int(cfg["input_dim"])
    potential_cache = _potential_cache_path(cfg)
    potential_cache.parent.mkdir(parents=True, exist_ok=True)

    source_seed = int(cfg.get("benchmark_source_seed", 0x000000))
    target_seed_v1 = int(cfg.get("benchmark_target_seed_v1", 0xBADBEEF))
    target_seed_v2 = int(cfg.get("benchmark_target_seed_v2", 0xC0FFEE))
    source_sampler = _random_gaussian_mixture(
        input_dim=input_dim,
        num_components=int(cfg.get("source_components", 3)),
        seed=source_seed,
        grid_spacing=float(cfg.get("grid_spacing", 1.0)),
        covariance_scale=float(cfg.get("covariance_scale", 0.4)),
    )
    target_sampler_v1 = _random_gaussian_mixture(
        input_dim=input_dim,
        num_components=int(cfg.get("target_components", 10)),
        seed=target_seed_v1,
        grid_spacing=float(cfg.get("grid_spacing", 1.0)),
        covariance_scale=float(cfg.get("covariance_scale", 0.4)),
    )
    target_sampler_v2 = _random_gaussian_mixture(
        input_dim=input_dim,
        num_components=int(cfg.get("target_components", 10)),
        seed=target_seed_v2,
        grid_spacing=float(cfg.get("grid_spacing", 1.0)),
        covariance_scale=float(cfg.get("covariance_scale", 0.4)),
    )

    if potential_cache.exists():
        payload = torch.load(potential_cache, map_location="cpu")
        left = _build_dense_icnn(cfg, input_dim=input_dim, constrained=True)
        right = _build_dense_icnn(cfg, input_dim=input_dim, constrained=True)
        left.load_state_dict(payload["left_state"])
        right.load_state_dict(payload["right_state"])
        _freeze(left)
        _freeze(right)
        summed = WeightedSumPotential([left, right], [1.0, 1.0])
        potential = ShiftScalePotential(
            summed,
            shift=payload["standardize_shift"],
            scale=float(payload["standardize_scale"]),
        )
        _freeze(potential)
        return source_sampler, potential

    left_cache = potential_cache.with_name(f"{potential_cache.stem}_v1.pt")
    right_cache = potential_cache.with_name(f"{potential_cache.stem}_v2.pt")
    left_payload = (
        torch.load(left_cache, map_location="cpu")
        if left_cache.exists()
        else _train_tw2_pair(source_sampler, target_sampler_v1, cfg, left_cache, seed_offset=11)
    )
    right_payload = (
        torch.load(right_cache, map_location="cpu")
        if right_cache.exists()
        else _train_tw2_pair(source_sampler, target_sampler_v2, cfg, right_cache, seed_offset=29)
    )

    left = _build_dense_icnn(cfg, input_dim=input_dim, constrained=True)
    right = _build_dense_icnn(cfg, input_dim=input_dim, constrained=True)
    left.load_state_dict(left_payload["forward_state"])
    right.load_state_dict(right_payload["forward_state"])
    _freeze(left)
    _freeze(right)

    summed = WeightedSumPotential([left, right], [1.0, 1.0])
    standardized = _standardize_potential(
        summed,
        source_sampler=source_sampler,
        estimate_size=int(cfg.get("standardize_samples", 2**14)),
        seed=int(cfg.get("standardize_seed", 0x000000)),
    )
    _freeze(standardized)

    torch.save(
        {
            "left_state": left.state_dict(),
            "right_state": right.state_dict(),
            "standardize_shift": standardized.shift,
            "standardize_scale": standardized.scale,
            "config": dict(cfg),
        },
        potential_cache,
    )
    return source_sampler, standardized


def _build_eval_split(
    source_sampler: GaussianMixtureSampler,
    target_sampler: PotentialPushforwardSampler,
    potential: nn.Module,
    num_samples: int,
    seed: int,
) -> dict[str, torch.Tensor]:
    generator = torch.Generator().manual_seed(seed)
    source = source_sampler.sample(num_samples, generator)
    target = target_sampler.sample(num_samples, generator)
    ground_truth_map = potential.gradient(source.clone(), create_graph=False).detach()
    return {
        "source": source,
        "target": target,
        "ground_truth_map": ground_truth_map,
    }


def build_paper_mix3to10_benchmark(config: Mapping[str, Any]) -> PaperBenchmarkBundle:
    """Build the paper's Mix3-to-Mix10 benchmark with a reconstructed tW2 benchmark map."""

    cfg = dict(config)
    construction = str(cfg.get("construction", "averaged_potential")).lower()
    if construction not in {"averaged_potential", "paper_reference"}:
        raise ValueError(f"Unsupported paper benchmark construction '{construction}'")

    source_sampler, ground_truth_potential = _load_ground_truth_potential(cfg)
    target_sampler = PotentialPushforwardSampler(source_sampler, ground_truth_potential)
    split_cache = _split_cache_path(cfg)
    split_cache.parent.mkdir(parents=True, exist_ok=True)

    if split_cache.exists():
        payload = torch.load(split_cache, map_location="cpu")
        val_dataset = TensorDictDataset(payload["val"])
        test_dataset = TensorDictDataset(payload["test"])
    else:
        payload = {
            "val": _build_eval_split(
                source_sampler,
                target_sampler,
                ground_truth_potential,
                num_samples=int(cfg["n_val"]),
                seed=int(cfg.get("seed", 1234)) + 22,
            ),
            "test": _build_eval_split(
                source_sampler,
                target_sampler,
                ground_truth_potential,
                num_samples=int(cfg["n_test"]),
                seed=int(cfg.get("seed", 1234)) + 33,
            ),
            "config": dict(cfg),
        }
        torch.save(payload, split_cache)
        val_dataset = TensorDictDataset(payload["val"])
        test_dataset = TensorDictDataset(payload["test"])

    train_loader = SamplerBatchLoader(
        source_sampler=source_sampler,
        target_sampler=target_sampler,
        ground_truth_potential=ground_truth_potential,
        batch_size=int(cfg.get("batch_size", 1024)),
        steps_per_epoch=int(cfg.get("steps_per_epoch", cfg.get("n_train", 65536) // cfg.get("batch_size", 1024))),
        seed=int(cfg.get("seed", 1234)) + 11,
    )
    return PaperBenchmarkBundle(
        train_loader=train_loader,
        val=val_dataset,
        test=test_dataset,
        ground_truth_potential=ground_truth_potential,
        batch_size=int(cfg.get("batch_size", 1024)),
        num_workers=int(cfg.get("num_workers", 0)),
    )
