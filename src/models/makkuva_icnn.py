"""Makkuva-style ICNN potentials."""

from __future__ import annotations

from collections.abc import Sequence

import torch
import torch.nn.functional as F
from torch import nn


def _truncated_normal_(
    tensor: torch.Tensor,
    *,
    std: float,
    threshold: float = 2.0,
) -> None:
    with torch.no_grad():
        values = torch.randn_like(tensor)
        values = values.clamp_(-threshold, threshold)
        tensor.copy_(values * std)


class MakkuvaPositiveLinear(nn.Module):
    """Linear layer whose weight is treated as a convexity-constrained parameter."""

    def __init__(
        self,
        in_features: int,
        out_features: int,
        *,
        bias: bool = False,
        weights_init_std: float = 0.1,
    ) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        self.weight.be_positive = True  # type: ignore[attr-defined]
        self.bias = nn.Parameter(torch.zeros(out_features)) if bias else None
        _truncated_normal_(self.weight, std=weights_init_std)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return F.linear(inputs, self.weight, self.bias)


class MakkuvaInputConvexNeuralNetwork(nn.Module):
    """Input-convex network matching the Makkuva et al. layer pattern."""

    def __init__(
        self,
        input_dim: int,
        hidden_dims: Sequence[int],
        *,
        activation: str = "leaky_relu",
        negative_slope: float = 0.2,
        input_quadratic: float = 0.0,
        weights_init_std: float = 0.1,
    ) -> None:
        super().__init__()
        if not hidden_dims:
            raise ValueError("hidden_dims must contain at least one entry")
        activation = activation.lower()
        if activation not in {"leaky_relu", "relu"}:
            raise ValueError("MakkuvaInputConvexNeuralNetwork only supports convex-safe relu variants")

        self.input_dim = int(input_dim)
        self.hidden_dims = list(hidden_dims)
        self.activation_name = activation
        self.negative_slope = float(negative_slope)
        self.input_quadratic = float(input_quadratic)

        self.input_layers = nn.ModuleList()
        for hidden_dim in self.hidden_dims:
            layer = nn.Linear(self.input_dim, hidden_dim, bias=True)
            _truncated_normal_(layer.weight, std=weights_init_std)
            nn.init.zeros_(layer.bias)
            self.input_layers.append(layer)

        self.hidden_layers = nn.ModuleList()
        for in_dim, out_dim in zip(self.hidden_dims[:-1], self.hidden_dims[1:]):
            self.hidden_layers.append(
                MakkuvaPositiveLinear(
                    in_features=in_dim,
                    out_features=out_dim,
                    bias=False,
                    weights_init_std=weights_init_std,
                )
            )

        self.output_hidden = MakkuvaPositiveLinear(
            in_features=self.hidden_dims[-1],
            out_features=1,
            bias=False,
            weights_init_std=weights_init_std,
        )
        self.output_input = nn.Linear(self.input_dim, 1, bias=True)
        _truncated_normal_(self.output_input.weight, std=weights_init_std)
        nn.init.zeros_(self.output_input.bias)

    def _activate(self, inputs: torch.Tensor) -> torch.Tensor:
        if self.activation_name == "relu":
            return F.relu(inputs)
        return F.leaky_relu(inputs, negative_slope=self.negative_slope)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        hidden = self._activate(self.input_layers[0](inputs)).pow(2)
        for input_layer, hidden_layer in zip(self.input_layers[1:], self.hidden_layers):
            hidden = self._activate(hidden_layer(hidden) + input_layer(inputs))
        output = self.output_hidden(hidden) + self.output_input(inputs)
        if self.input_quadratic > 0.0:
            output = output + 0.5 * self.input_quadratic * inputs.pow(2).sum(dim=-1, keepdim=True)
        return output

    def gradient(self, inputs: torch.Tensor, create_graph: bool = False) -> torch.Tensor:
        inputs = inputs.requires_grad_(True)
        values = self.forward(inputs)
        return torch.autograd.grad(values.sum(), inputs, create_graph=create_graph)[0]

    def positive_parameters(self) -> list[nn.Parameter]:
        parameters = [layer.weight for layer in self.hidden_layers]
        parameters.append(self.output_hidden.weight)
        return parameters

    def convexify(self) -> None:
        with torch.no_grad():
            for parameter in self.positive_parameters():
                parameter.clamp_(min=0.0)

    def negative_weight_penalty(self) -> torch.Tensor:
        penalty = self.output_hidden.weight.new_tensor(0.0)
        for parameter in self.positive_parameters():
            penalty = penalty + torch.relu(-parameter).pow(2).sum()
        return penalty
