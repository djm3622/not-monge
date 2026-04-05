"""Flow-based model components for OT baselines."""

from __future__ import annotations

from typing import Any, Iterable, Sequence

import torch
from torch import nn

from src.models.common import get_activation


class TimeConditionedVectorField(nn.Module):
    """MLP vector field conditioned on continuous time."""

    def __init__(
        self,
        input_dim: int,
        hidden_dims: Sequence[int],
        activation: str = "silu",
        layer_norm: bool = False,
    ) -> None:
        super().__init__()
        dims = [input_dim + 1, *hidden_dims, input_dim]
        layers: list[nn.Module] = []
        for in_dim, out_dim in zip(dims[:-2], dims[1:-1]):
            layers.append(nn.Linear(in_dim, out_dim))
            if layer_norm:
                layers.append(nn.LayerNorm(out_dim))
            layers.append(get_activation(activation))
        layers.append(nn.Linear(dims[-2], dims[-1]))
        self.network = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        if t.ndim == 0:
            time = t.expand(x.shape[0]).unsqueeze(-1)
        elif t.ndim == 1:
            time = t.unsqueeze(-1)
        else:
            time = t.reshape(x.shape[0], 1)
        return self.network(torch.cat([x, time.to(dtype=x.dtype)], dim=-1))


def build_vector_field(
    config: dict[str, Any] | Iterable[tuple[str, object]],
) -> TimeConditionedVectorField:
    """Instantiate a time-conditioned vector field from config."""
    cfg = dict(config)
    return TimeConditionedVectorField(
        input_dim=int(cfg["input_dim"]),
        hidden_dims=list(cfg["hidden_dims"]),
        activation=str(cfg.get("activation", "silu")),
        layer_norm=bool(cfg.get("layer_norm", False)),
    )
