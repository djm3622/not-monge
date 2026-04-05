"""Data helpers used by training and reference solvers."""

from __future__ import annotations

from typing import Iterable, Sequence

import torch
from torch.utils.data import DataLoader


def collect_loader_tensors(
    loader: DataLoader[dict[str, torch.Tensor]],
    keys: Sequence[str],
    max_items: int | None = None,
) -> dict[str, torch.Tensor]:
    """Collect a capped number of samples from a loader."""
    chunks = {key: [] for key in keys}
    collected = 0
    for batch in loader:
        batch_size = next(iter(batch.values())).shape[0]
        take = batch_size
        if max_items is not None:
            remaining = max_items - collected
            if remaining <= 0:
                break
            take = min(batch_size, remaining)
        for key in keys:
            chunks[key].append(batch[key][:take].detach().cpu())
        collected += take
        if max_items is not None and collected >= max_items:
            break
    return {key: torch.cat(value, dim=0) for key, value in chunks.items()}


def maybe_override_batch_size(config: dict, batch_size: int | None) -> dict:
    """Override a dataset batch size when fairness settings request it."""
    updated = dict(config)
    if batch_size is not None and "batch_size" in updated:
        updated["batch_size"] = int(batch_size)
    return updated


def split_tensor_dict(tensors: dict[str, torch.Tensor], batch_size: int) -> Iterable[dict[str, torch.Tensor]]:
    """Yield mini-batches from a dictionary of tensors."""
    total = next(iter(tensors.values())).shape[0]
    for start in range(0, total, batch_size):
        end = min(total, start + batch_size)
        yield {key: value[start:end] for key, value in tensors.items()}
