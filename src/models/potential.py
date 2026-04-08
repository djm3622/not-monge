"""Potential networks for quadratic-cost OT."""

from __future__ import annotations

from typing import Iterable, Sequence

import torch
import torch.nn.functional as F
from torch import nn

from src.models.common import get_activation
from src.models.makkuva_icnn import MakkuvaInputConvexNeuralNetwork


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

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        init_std: float = 0.02,
    ) -> None:
        super().__init__()
        self.weight_param = nn.Parameter(torch.empty(out_features, in_features))
        self.bias = nn.Parameter(torch.zeros(out_features)) if bias else None
        nn.init.normal_(self.weight_param, mean=0.0, std=init_std)

    @property
    def weight(self) -> torch.Tensor:
        return self.weight_param.clamp_min(0.0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.linear(x, self.weight, self.bias)


class QuadraticSkip(nn.Module):
    """Input-quadratic skip layer used by DenseICNN-style architectures."""

    def __init__(
        self,
        in_features: int,
        out_features: int,
        rank: int = 1,
        bias: bool = True,
        init_std: float = 0.01,
    ) -> None:
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.rank = rank
        self.linear_weight = nn.Parameter(torch.empty(out_features, in_features))
        self.quadratic_factors = nn.Parameter(torch.empty(out_features, rank, in_features))
        self.bias = nn.Parameter(torch.zeros(out_features)) if bias else None
        nn.init.normal_(self.linear_weight, mean=0.0, std=init_std)
        nn.init.normal_(self.quadratic_factors, mean=0.0, std=init_std)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        linear = F.linear(x, self.linear_weight, self.bias)
        projected = torch.einsum("bi,ori->bor", x, self.quadratic_factors)
        quadratic = projected.pow(2).sum(dim=-1)
        return linear + quadratic


class DenseInputConvexNeuralNetwork(nn.Module):
    """Dense ICNN with input-quadratic skip connections."""

    def __init__(
        self,
        input_dim: int,
        hidden_dims: Sequence[int],
        rank: int = 1,
        activation: str = "celu",
        dropout: float = 0.0,
        strong_convexity: float = 0.0,
        identity_quadratic: float = 0.0,
        constrain_convex: bool = True,
        weights_init_std: float = 0.01,
    ) -> None:
        super().__init__()
        if not hidden_dims:
            raise ValueError("hidden_dims must contain at least one entry")
        self.input_dim = input_dim
        self.hidden_dims = list(hidden_dims)
        self.rank = rank
        self.dropout = float(dropout)
        self.strong_convexity = float(strong_convexity)
        self.identity_quadratic = float(identity_quadratic)
        self.activation_name = activation
        self.constrain_convex = bool(constrain_convex)
        self.weights_init_std = float(weights_init_std)

        self.quadratic_layers = nn.ModuleList(
            [
                QuadraticSkip(
                    input_dim,
                    width,
                    rank=rank,
                    bias=True,
                    init_std=self.weights_init_std,
                )
                for width in self.hidden_dims
            ]
        )
        self.hidden_layers = nn.ModuleList()
        for in_dim, out_dim in zip(self.hidden_dims[:-1], self.hidden_dims[1:]):
            if constrain_convex:
                layer = NonNegativeLinear(in_dim, out_dim, bias=False, init_std=self.weights_init_std)
            else:
                layer = nn.Linear(in_dim, out_dim, bias=False)
                nn.init.normal_(layer.weight, mean=0.0, std=self.weights_init_std)
            self.hidden_layers.append(layer)

        if constrain_convex:
            self.output_layer: nn.Module = NonNegativeLinear(
                self.hidden_dims[-1],
                1,
                bias=False,
                init_std=self.weights_init_std,
            )
        else:
            output = nn.Linear(self.hidden_dims[-1], 1, bias=False)
            nn.init.normal_(output.weight, mean=0.0, std=self.weights_init_std)
            self.output_layer = output

    def _activate(self, x: torch.Tensor) -> torch.Tensor:
        if self.activation_name == "celu":
            return torch.celu(x)
        return get_activation(self.activation_name)(x)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        output = self.quadratic_layers[0](x)
        if self.dropout > 0.0:
            output = F.dropout(output, p=self.dropout, training=self.training)
        for skip_layer, hidden_layer in zip(self.quadratic_layers[1:], self.hidden_layers):
            output = hidden_layer(output) + skip_layer(x)
            output = self._activate(output)
            if self.dropout > 0.0:
                output = F.dropout(output, p=self.dropout, training=self.training)

        potential = self.output_layer(output)
        quadratic_scale = self.strong_convexity + self.identity_quadratic
        if quadratic_scale != 0.0:
            potential = potential + 0.5 * quadratic_scale * x.pow(2).sum(dim=-1, keepdim=True)
        return potential

    def gradient(self, x: torch.Tensor, create_graph: bool = False) -> torch.Tensor:
        x = x.requires_grad_(True)
        potential = self.forward(x)
        return torch.autograd.grad(
            potential.sum(),
            x,
            create_graph=create_graph,
        )[0]

    def convexify(self) -> None:
        if not self.constrain_convex:
            return
        for layer in self.hidden_layers:
            if isinstance(layer, NonNegativeLinear):
                layer.weight_param.data.clamp_(min=0.0)
        if isinstance(self.output_layer, NonNegativeLinear):
            self.output_layer.weight_param.data.clamp_(min=0.0)


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


def _scale_module_parameters(module: nn.Module, factor: float) -> nn.Module:
    """Scale learned parameters to start close to the identity map."""
    if factor == 1.0:
        return module
    with torch.no_grad():
        for parameter in module.parameters():
            parameter.mul_(factor)
    return module


def build_potential(config: dict | Iterable[tuple[str, object]]) -> nn.Module:
    """Instantiate a potential network from config."""
    cfg = dict(config)
    kind = str(cfg.get("kind", cfg.get("name", "mlp"))).lower()
    identity_init = bool(cfg.get("initialize_identity", False))
    identity_scale = float(cfg.get("identity_init_scale", 1.0e-2))
    if "makkuva_icnn" in kind:
        module = MakkuvaInputConvexNeuralNetwork(
            input_dim=int(cfg["input_dim"]),
            hidden_dims=list(cfg["hidden_dims"]),
            activation=str(cfg.get("activation", "leaky_relu")),
            negative_slope=float(cfg.get("negative_slope", 0.2)),
            input_quadratic=float(cfg.get("input_quadratic", 0.0)),
            weights_init_std=float(cfg.get("weights_init_std", 0.1)),
        )
        return _scale_module_parameters(module, identity_scale if identity_init else 1.0)
    if "denseicnn_u" in kind or "dense_icnn_u" in kind:
        module = DenseInputConvexNeuralNetwork(
            input_dim=int(cfg["input_dim"]),
            hidden_dims=list(cfg["hidden_dims"]),
            rank=int(cfg.get("rank", 1)),
            activation=str(cfg.get("activation", "celu")),
            dropout=float(cfg.get("dropout", 0.0)),
            strong_convexity=float(cfg.get("strong_convexity", 0.0)),
            identity_quadratic=float(cfg.get("identity_quadratic", 0.0)),
            constrain_convex=False,
            weights_init_std=float(cfg.get("weights_init_std", 0.01)),
        )
        return _scale_module_parameters(module, identity_scale if identity_init else 1.0)
    if "denseicnn" in kind or "dense_icnn" in kind:
        module = DenseInputConvexNeuralNetwork(
            input_dim=int(cfg["input_dim"]),
            hidden_dims=list(cfg["hidden_dims"]),
            rank=int(cfg.get("rank", 1)),
            activation=str(cfg.get("activation", "celu")),
            dropout=float(cfg.get("dropout", 0.0)),
            strong_convexity=float(cfg.get("strong_convexity", 0.0)),
            identity_quadratic=float(cfg.get("identity_quadratic", 0.0)),
            constrain_convex=True,
            weights_init_std=float(cfg.get("weights_init_std", 0.01)),
        )
        return _scale_module_parameters(module, identity_scale if identity_init else 1.0)
    if "icnn" in kind:
        module = InputConvexNeuralNetwork(
            input_dim=int(cfg["input_dim"]),
            hidden_dims=list(cfg["hidden_dims"]),
            activation=str(cfg.get("activation", "softplus")),
            strong_convexity=float(cfg.get("strong_convexity", 0.0)),
        )
        return _scale_module_parameters(module, identity_scale if identity_init else 1.0)
    module = PotentialMLP(
        input_dim=int(cfg["input_dim"]),
        hidden_dims=list(cfg["hidden_dims"]),
        activation=str(cfg.get("activation", "silu")),
        layer_norm=bool(cfg.get("layer_norm", False)),
    )
    return _scale_module_parameters(module, identity_scale if identity_init else 1.0)
