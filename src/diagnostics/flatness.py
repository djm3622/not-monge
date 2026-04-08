"""Minimal semi-dual flatness diagnostics."""

from __future__ import annotations

import copy
from typing import Callable, Mapping

import torch
from torch import nn


def empirical_semidual_objective(
    map_fn: Callable[[torch.Tensor], torch.Tensor],
    potential: nn.Module,
    source: torch.Tensor,
    target: torch.Tensor,
) -> float:
    """Compute the empirical semi-dual objective for a fixed map and potential."""
    transported = map_fn(source)
    cost = 0.5 * (source - transported).pow(2).sum(dim=-1).mean()
    potential_target = potential(target).view(-1).mean()
    potential_transported = potential(transported).view(-1).mean()
    return float((cost + potential_target - potential_transported).detach())


def make_noisy_potential_copy(
    potential: nn.Module,
    *,
    noise_scale: float,
    seed: int,
) -> nn.Module:
    """Return a noisy copy of a potential without mutating the original."""
    clone = copy.deepcopy(potential)
    generator = torch.Generator().manual_seed(seed)
    with torch.no_grad():
        for parameter in clone.parameters():
            if parameter.numel() == 0:
                continue
            noise = torch.randn(
                parameter.shape,
                generator=generator,
                device=parameter.device,
                dtype=parameter.dtype,
            )
            parameter.add_(noise_scale * noise)
    return clone


def summarize_flatness(
    values: Mapping[str, float],
    *,
    final_key: str = "final",
) -> dict[str, float]:
    """Summarize sensitivity of the empirical semi-dual objective to potential variants."""
    if final_key not in values:
        raise KeyError(f"final_key '{final_key}' missing from flatness values")
    tensor = torch.tensor(list(values.values()), dtype=torch.float32)
    final_value = float(values[final_key])
    gaps = torch.tensor([abs(value - final_value) for key, value in values.items() if key != final_key], dtype=torch.float32)
    mean_abs_gap = 0.0 if gaps.numel() == 0 else float(gaps.mean())
    return {
        "flatness_std_F": float(tensor.std(unbiased=False)),
        "flatness_range_F": float(tensor.max() - tensor.min()),
        "flatness_mean_abs_gap_to_final": mean_abs_gap,
    }
