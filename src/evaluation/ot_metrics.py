"""Metrics for transport-map recovery."""

from __future__ import annotations

from typing import Callable

import numpy as np
import torch
from scipy.optimize import linear_sum_assignment


def map_l2_error(predicted: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Root-mean-square error between predicted and target maps."""
    return (predicted - target).pow(2).mean().sqrt()


def l2_unexplained_variance_percentage(
    predicted: torch.Tensor,
    target: torch.Tensor,
    reference_samples: torch.Tensor,
) -> float:
    """Paper-style L2-UVP score in percent."""
    numerator = (predicted - target).pow(2).sum(dim=-1).mean().detach()
    centered = (reference_samples - reference_samples.mean(dim=0, keepdim=True)).detach()
    denominator = centered.pow(2).sum(dim=-1).mean().clamp_min(1e-8)
    return float(100.0 * numerator / denominator)


def transport_cosine_similarity(
    predicted: torch.Tensor,
    target: torch.Tensor,
    source: torch.Tensor,
) -> float:
    """Paper-style transport cosine normalized by benchmark transport cost."""
    predicted_delta = predicted - source
    target_delta = target - source
    predicted_energy = predicted_delta.pow(2).sum(dim=-1).mean().detach()
    target_cost = (0.5 * target_delta.pow(2).sum(dim=-1).mean()).detach()
    if float(predicted_energy) == 0.0 and float(target_cost) == 0.0:
        return 1.0
    if float(predicted_energy) == 0.0 or float(target_cost) == 0.0:
        return 0.0
    numerator = (target_delta * predicted_delta).sum(dim=-1).mean().detach()
    denominator = torch.sqrt((2.0 * target_cost).clamp_min(1e-8) * predicted_energy.clamp_min(1e-8)).detach()
    return float(numerator / denominator)


def empirical_w2_distance(
    predicted: torch.Tensor,
    target: torch.Tensor,
    max_samples: int = 1024,
) -> float:
    """Approximate empirical W2 with exact assignment on a capped sample size."""
    if predicted.shape[0] > max_samples:
        indices = torch.randperm(predicted.shape[0], device=predicted.device)[:max_samples]
        predicted = predicted[indices]
        target = target[indices]
    cost = torch.cdist(predicted, target).pow(2).detach().cpu().numpy()
    row_ind, col_ind = linear_sum_assignment(cost)
    return float(np.sqrt(cost[row_ind, col_ind].mean()))


def maximum_mean_discrepancy(
    samples_a: torch.Tensor,
    samples_b: torch.Tensor,
    bandwidth: float | None = None,
) -> float:
    """RBF-kernel MMD."""
    combined = torch.cat([samples_a, samples_b], dim=0)
    pairwise = torch.cdist(combined, combined).pow(2)
    if bandwidth is None:
        bandwidth = float(pairwise.median().clamp_min(1e-6).detach())
    gamma = 1.0 / (2.0 * bandwidth)
    kernel = torch.exp(-gamma * pairwise)
    n = samples_a.shape[0]
    xx = kernel[:n, :n].mean()
    yy = kernel[n:, n:].mean()
    xy = kernel[:n, n:].mean()
    return float((xx + yy - 2.0 * xy).detach())


def batch_jacobian(model: Callable[[torch.Tensor], torch.Tensor], inputs: torch.Tensor) -> torch.Tensor:
    """Compute per-sample Jacobians."""
    inputs = inputs.clone().requires_grad_(True)
    outputs = model(inputs)
    jacobian_rows = []
    for column in range(outputs.shape[-1]):
        gradient = torch.autograd.grad(
            outputs[:, column].sum(),
            inputs,
            retain_graph=column < outputs.shape[-1] - 1,
        )[0]
        jacobian_rows.append(gradient.unsqueeze(1))
    return torch.cat(jacobian_rows, dim=1)


def gradient_error(
    predicted_map: Callable[[torch.Tensor], torch.Tensor],
    reference_map: Callable[[torch.Tensor], torch.Tensor],
    inputs: torch.Tensor,
) -> float:
    """Frobenius error between Jacobians of two maps."""
    predicted_jac = batch_jacobian(predicted_map, inputs)
    reference_jac = batch_jacobian(reference_map, inputs)
    return float((predicted_jac - reference_jac).pow(2).mean().sqrt().detach())


def saddle_residual(
    forward_map: Callable[[torch.Tensor], torch.Tensor],
    inverse_map: Callable[[torch.Tensor], torch.Tensor],
    targets: torch.Tensor,
) -> float:
    """Root-mean-square inverse-response residual ||T(\hat{x}(y)) - y||."""
    inverse = inverse_map(targets)
    reconstructed = forward_map(inverse)
    return float((reconstructed - targets).pow(2).mean().sqrt().detach())
