"""Transport map parameterizations."""

from __future__ import annotations

from typing import Iterable, Sequence

import torch
from torch import nn

from src.models.common import get_activation


class ResidualMLPBlock(nn.Module):
    """Residual block used by the transport network."""

    def __init__(
        self,
        hidden_dim: int,
        activation: str,
        dropout: float = 0.0,
        layer_norm: bool = False,
    ) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(hidden_dim) if layer_norm else nn.Identity()
        self.block = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            get_activation(activation),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.activation = get_activation(activation)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = self.norm(x)
        x = self.block(x)
        return self.activation(x + residual)


class OTMapNetwork(nn.Module):
    """MLP transport map with optional residual hidden blocks."""

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        hidden_dims: Sequence[int],
        activation: str = "silu",
        dropout: float = 0.0,
        residual: bool = True,
        layer_norm: bool = False,
        output_residual: bool = False,
        zero_init_output: bool = False,
    ) -> None:
        super().__init__()
        if not hidden_dims:
            raise ValueError("hidden_dims must contain at least one entry")
        if output_residual and input_dim != output_dim:
            raise ValueError("output_residual requires matching input_dim and output_dim")
        first_hidden, *tail = list(hidden_dims)
        self.input = nn.Sequential(
            nn.Linear(input_dim, first_hidden),
            get_activation(activation),
        )
        blocks: list[nn.Module] = []
        in_dim = first_hidden
        for hidden_dim in tail:
            blocks.append(nn.Linear(in_dim, hidden_dim))
            blocks.append(get_activation(activation))
            if layer_norm:
                blocks.append(nn.LayerNorm(hidden_dim))
            if dropout > 0.0:
                blocks.append(nn.Dropout(dropout))
            if residual and hidden_dim == in_dim:
                blocks.append(ResidualMLPBlock(hidden_dim, activation, dropout, layer_norm))
            in_dim = hidden_dim
        self.hidden = nn.Sequential(*blocks) if blocks else nn.Identity()
        self.output = nn.Linear(in_dim, output_dim)
        self.output_residual = bool(output_residual)
        if zero_init_output:
            nn.init.zeros_(self.output.weight)
            nn.init.zeros_(self.output.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = self.input(x)
        x = self.hidden(x)
        output = self.output(x)
        if self.output_residual:
            output = output + residual
        return output


def build_ot_map(config: dict | Iterable[tuple[str, object]]) -> OTMapNetwork:
    """Instantiate a transport map from a config mapping."""
    cfg = dict(config)
    return OTMapNetwork(
        input_dim=int(cfg["input_dim"]),
        output_dim=int(cfg.get("output_dim", cfg["input_dim"])),
        hidden_dims=list(cfg["hidden_dims"]),
        activation=str(cfg.get("activation", "silu")),
        dropout=float(cfg.get("dropout", 0.0)),
        residual=bool(cfg.get("residual", True)),
        layer_norm=bool(cfg.get("layer_norm", False)),
        output_residual=bool(cfg.get("output_residual", False)),
        zero_init_output=bool(cfg.get("zero_init_output", False)),
    )
