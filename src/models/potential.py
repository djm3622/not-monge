"""Potential networks for quadratic-cost OT."""

from __future__ import annotations

from typing import Iterable, Sequence

import torch
import torch.nn.functional as F
from torch import nn

from src.models.common import get_activation


class PotentialMLP(nn.Module):
    """Standard unconstrained scalar potential."""

    def __init__(
        self,
        input_dim: int,
        hidden_dims: Sequence[int],
        activation: str = "silu",
        layer_norm: bool = False,
    ) -> None:
        super().__init__()
        dims = [input_dim, *hidden_dims, 1]
        layers: list[nn.Module] = []
        for in_dim, out_dim in zip(dims[:-2], dims[1:-1]):
            layers.append(nn.Linear(in_dim, out_dim))
            if layer_norm:
                layers.append(nn.LayerNorm(out_dim))
            layers.append(get_activation(activation))
        layers.append(nn.Linear(dims[-2], dims[-1]))
        self.network = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)

    def gradient(self, x: torch.Tensor, create_graph: bool = False) -> torch.Tensor:
        x = x.requires_grad_(True)
        potential = self.forward(x)
        return torch.autograd.grad(
            potential.sum(),
            x,
            create_graph=create_graph,
        )[0]


class NonNegativeLinear(nn.Module):
    """Linear layer whose weights are constrained to be non-negative."""

    def __init__(self, in_features: int, out_features: int, bias: bool = True) -> None:
        super().__init__()
        self.raw_weight = nn.Parameter(torch.empty(out_features, in_features))
        self.bias = nn.Parameter(torch.zeros(out_features)) if bias else None
        nn.init.xavier_uniform_(self.raw_weight)

    @property
    def weight(self) -> torch.Tensor:
        return F.softplus(self.raw_weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.linear(x, self.weight, self.bias)


class InputConvexNeuralNetwork(nn.Module):
    """ICNN with non-negative hidden-to-hidden weights and strong convexity."""

    def __init__(
        self,
        input_dim: int,
        hidden_dims: Sequence[int],
        activation: str = "softplus",
        strong_convexity: float = 0.0,
    ) -> None:
        super().__init__()
        if not hidden_dims:
            raise ValueError("hidden_dims must contain at least one entry")
        self.input_dim = input_dim
        self.strong_convexity = strong_convexity
        self.activation = get_activation(activation)
        self.x_layers = nn.ModuleList()
        self.z_layers = nn.ModuleList()
        prev_hidden = 0
        for index, hidden_dim in enumerate(hidden_dims):
            self.x_layers.append(nn.Linear(input_dim, hidden_dim))
            if index > 0:
                self.z_layers.append(NonNegativeLinear(prev_hidden, hidden_dim, bias=False))
            prev_hidden = hidden_dim
        self.output_x = nn.Linear(input_dim, 1)
        self.output_z = NonNegativeLinear(hidden_dims[-1], 1, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.activation(self.x_layers[0](x))
        for x_layer, z_layer in zip(self.x_layers[1:], self.z_layers):
            z = self.activation(x_layer(x) + z_layer(z))
        output = self.output_x(x) + self.output_z(z)
        if self.strong_convexity > 0.0:
            output = output + 0.5 * self.strong_convexity * x.pow(2).sum(dim=-1, keepdim=True)
        return output

    def gradient(self, x: torch.Tensor, create_graph: bool = False) -> torch.Tensor:
        x = x.requires_grad_(True)
        potential = self.forward(x)
        return torch.autograd.grad(
            potential.sum(),
            x,
            create_graph=create_graph,
        )[0]


def build_potential(config: dict | Iterable[tuple[str, object]]) -> nn.Module:
    """Instantiate a potential network from config."""
    cfg = dict(config)
    kind = str(cfg.get("kind", cfg.get("name", "mlp"))).lower()
    if "icnn" in kind:
        return InputConvexNeuralNetwork(
            input_dim=int(cfg["input_dim"]),
            hidden_dims=list(cfg["hidden_dims"]),
            activation=str(cfg.get("activation", "softplus")),
            strong_convexity=float(cfg.get("strong_convexity", 0.0)),
        )
    return PotentialMLP(
        input_dim=int(cfg["input_dim"]),
        hidden_dims=list(cfg["hidden_dims"]),
        activation=str(cfg.get("activation", "silu")),
        layer_norm=bool(cfg.get("layer_norm", False)),
    )
