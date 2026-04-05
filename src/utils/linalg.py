"""Linear algebra helpers for OT baselines."""

from __future__ import annotations

import torch


def matrix_symmetric_eig(
    matrix: torch.Tensor,
    jitter: float = 1.0e-6,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Safely eigendecompose a symmetric matrix."""
    if matrix.shape[-1] != matrix.shape[-2]:
        raise ValueError("matrix must be square")
    identity = torch.eye(matrix.shape[-1], device=matrix.device, dtype=matrix.dtype)
    stabilized = 0.5 * (matrix + matrix.transpose(-1, -2)) + jitter * identity
    eigenvalues, eigenvectors = torch.linalg.eigh(stabilized)
    return eigenvalues.clamp_min(jitter), eigenvectors


def symmetric_matrix_square_root(
    matrix: torch.Tensor,
    inverse: bool = False,
    jitter: float = 1.0e-6,
) -> torch.Tensor:
    """Compute a symmetric matrix square root or inverse square root."""
    eigenvalues, eigenvectors = matrix_symmetric_eig(matrix, jitter=jitter)
    powers = eigenvalues.rsqrt() if inverse else eigenvalues.sqrt()
    return (eigenvectors * powers.unsqueeze(0)) @ eigenvectors.transpose(-1, -2)


def gaussian_ot_linear_map(
    source_covariance: torch.Tensor,
    target_covariance: torch.Tensor,
    jitter: float = 1.0e-6,
) -> torch.Tensor:
    """Closed-form linear Gaussian OT map."""
    source_sqrt = symmetric_matrix_square_root(source_covariance, inverse=False, jitter=jitter)
    source_inv_sqrt = symmetric_matrix_square_root(source_covariance, inverse=True, jitter=jitter)
    middle = source_sqrt @ target_covariance @ source_sqrt
    return source_inv_sqrt @ symmetric_matrix_square_root(middle, inverse=False, jitter=jitter) @ source_inv_sqrt
