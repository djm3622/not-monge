"""Metrics for c-concavity and convex-envelope analysis."""

from __future__ import annotations

from typing import Callable

import torch


def quadratic_pairwise_cost(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """Pairwise quadratic costs."""
    return 0.5 * torch.cdist(x, y).pow(2)


def numerical_c_transform(
    source_points: torch.Tensor,
    target_points: torch.Tensor,
    potential_values: torch.Tensor,
) -> torch.Tensor:
    """Discrete c-transform on sample clouds."""
    values = potential_values.view(-1, 1)
    costs = quadratic_pairwise_cost(source_points, target_points)
    return (costs - values).amin(dim=0)


def c_concave_envelope(
    support_points: torch.Tensor,
    potential_values: torch.Tensor,
) -> torch.Tensor:
    """Compute the discrete c-concave envelope via a double c-transform."""
    first_transform = numerical_c_transform(support_points, support_points, potential_values)
    return numerical_c_transform(support_points, support_points, first_transform)


def envelope_gap(
    support_points: torch.Tensor,
    potential_values: torch.Tensor,
) -> dict[str, float]:
    """Measure the discrepancy between a function and its c-concave envelope."""
    envelope = c_concave_envelope(support_points, potential_values)
    gap = potential_values.flatten() - envelope.flatten()
    positive_gap = torch.relu(gap)
    return {
        "envelope_gap/mean": float(positive_gap.mean()),
        "envelope_gap/max": float(positive_gap.max()),
    }


def convexity_violation(
    potential: Callable[[torch.Tensor], torch.Tensor],
    points: torch.Tensor,
    num_pairs: int = 256,
) -> dict[str, float]:
    """Estimate Jensen-inequality violations."""
    if points.shape[0] < 2:
        return {"convexity_violation/mean": 0.0, "convexity_violation/max": 0.0}
    indices_a = torch.randint(points.shape[0], (num_pairs,), device=points.device)
    indices_b = torch.randint(points.shape[0], (num_pairs,), device=points.device)
    lambdas = torch.rand(num_pairs, 1, device=points.device)
    point_a = points[indices_a]
    point_b = points[indices_b]
    midpoint = lambdas * point_a + (1.0 - lambdas) * point_b
    potential_mid = potential(midpoint).view(-1)
    upper_bound = (
        lambdas.view(-1) * potential(point_a).view(-1)
        + (1.0 - lambdas.view(-1)) * potential(point_b).view(-1)
    )
    violation = torch.relu(potential_mid - upper_bound)
    return {
        "convexity_violation/mean": float(violation.mean().detach()),
        "convexity_violation/max": float(violation.max().detach()),
    }


def hessian_spectrum(
    potential: Callable[[torch.Tensor], torch.Tensor],
    points: torch.Tensor,
    max_points: int = 8,
) -> dict[str, float]:
    """Compute aggregate Hessian eigenvalue statistics on a small sample."""
    points = points[:max_points].detach()
    eigenvalues = []
    for point in points:
        point = point.clone().requires_grad_(True)
        hessian = torch.autograd.functional.hessian(lambda p: potential(p.unsqueeze(0)).sum(), point)
        eigenvalues.append(torch.linalg.eigvalsh(hessian).detach())
    stacked = torch.cat(eigenvalues)
    return {
        "hessian/min_eig": float(stacked.min()),
        "hessian/max_eig": float(stacked.max()),
        "hessian/mean_eig": float(stacked.mean()),
    }
