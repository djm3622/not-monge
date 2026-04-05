"""Learning-rate schedulers."""

from __future__ import annotations

from collections.abc import Sequence

import torch


def build_one_cycle_schedulers(
    optimizers: Sequence[torch.optim.Optimizer],
    total_steps: int,
    max_lrs: Sequence[float],
    pct_start: float = 0.1,
    div_factor: float = 25.0,
    final_div_factor: float = 1_000.0,
) -> list[torch.optim.lr_scheduler.OneCycleLR]:
    """Create one-cycle schedulers for a list of optimizers."""
    if total_steps <= 0:
        raise ValueError("total_steps must be positive")
    schedulers: list[torch.optim.lr_scheduler.OneCycleLR] = []
    for optimizer, max_lr in zip(optimizers, max_lrs):
        schedulers.append(
            torch.optim.lr_scheduler.OneCycleLR(
                optimizer,
                max_lr=max_lr,
                total_steps=total_steps,
                pct_start=pct_start,
                div_factor=div_factor,
                final_div_factor=final_div_factor,
            )
        )
    return schedulers
