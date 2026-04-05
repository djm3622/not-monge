"""Losses used across OT and diffusion experiments."""

from __future__ import annotations

from typing import Callable

import torch
import torch.nn.functional as F


def quadratic_cost(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """Per-sample quadratic transport cost."""
    return 0.5 * (x - y).pow(2).sum(dim=-1)


def minimax_potential_objective(
    potential_values_real: torch.Tensor,
    potential_values_fake: torch.Tensor,
) -> torch.Tensor:
    """Potential maximization objective."""
    return potential_values_real.mean() - potential_values_fake.mean()


def minimax_transport_objective(
    source: torch.Tensor,
    transported: torch.Tensor,
    potential_values_fake: torch.Tensor,
    cost_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor] = quadratic_cost,
) -> torch.Tensor:
    """Transport minimization objective."""
    return cost_fn(source, transported).mean() - potential_values_fake.mean()


def ddpm_noise_prediction_loss(
    prediction: torch.Tensor,
    target_noise: torch.Tensor,
) -> torch.Tensor:
    """Standard DDPM MSE objective."""
    return F.mse_loss(prediction, target_noise)
