"""GroupSort-based Lipschitz networks for W1 baselines."""

from __future__ import annotations

from collections.abc import Sequence

import torch
import torch.nn.functional as F
from torch import nn


class GroupSort(nn.Module):
    """GroupSort activation."""

    def __init__(self, group_size: int = 2) -> None:
        super().__init__()
        self.group_size = group_size

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.shape[-1] % self.group_size != 0:
            raise ValueError("features must be divisible by group_size for GroupSort")
        original_shape = x.shape
        x = x.view(*x.shape[:-1], x.shape[-1] // self.group_size, self.group_size)
        x, _ = torch.sort(x, dim=-1)
        return x.view(*original_shape)


class GroupSortMLP(nn.Module):
    """Spectrally-normalized MLP with GroupSort activations."""

    def __init__(
        self,
        input_dim: int,
        hidden_dims: Sequence[int],
        output_dim: int = 1,
        group_size: int = 2,
    ) -> None:
        super().__init__()
        dims = [input_dim, *hidden_dims, output_dim]
        layers: list[nn.Module] = []
        for index, (in_dim, out_dim) in enumerate(zip(dims[:-1], dims[1:])):
            linear = nn.utils.parametrizations.spectral_norm(nn.Linear(in_dim, out_dim))
            layers.append(linear)
            if index < len(dims) - 2:
                layers.append(GroupSort(group_size=group_size))
        self.network = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)
