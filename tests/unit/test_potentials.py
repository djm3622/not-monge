from __future__ import annotations

import pytest
import torch

from src.models.potential import (
    InputConvexNeuralNetwork,
    NonNegativeLinear,
    PotentialMLP,
    build_potential,
)

pytestmark = pytest.mark.unit


@pytest.mark.parametrize(
    ("builder", "config"),
    [
        (
            PotentialMLP,
            {"input_dim": 2, "hidden_dims": [8, 8], "activation": "silu", "layer_norm": False},
        ),
        (
            InputConvexNeuralNetwork,
            {"input_dim": 2, "hidden_dims": [8, 8], "activation": "softplus", "strong_convexity": 0.1},
        ),
    ],
)
def test_potential_outputs_scalar_and_is_finite(
    builder: type[torch.nn.Module],
    config: dict[str, object],
) -> None:
    potential = builder(**config)
    inputs = torch.randn(6, int(config["input_dim"]))
    outputs = potential(inputs)
    assert outputs.shape == (6, 1)
    assert torch.isfinite(outputs).all()


@pytest.mark.parametrize(
    ("builder", "config"),
    [
        (
            PotentialMLP,
            {"input_dim": 2, "hidden_dims": [8, 8], "activation": "silu", "layer_norm": False},
        ),
        (
            InputConvexNeuralNetwork,
            {"input_dim": 2, "hidden_dims": [8, 8], "activation": "softplus", "strong_convexity": 0.1},
        ),
    ],
)
def test_potential_gradients_exist_and_match_input_shape(
    builder: type[torch.nn.Module],
    config: dict[str, object],
) -> None:
    potential = builder(**config)
    inputs = torch.randn(4, int(config["input_dim"]))
    gradients = potential.gradient(inputs, create_graph=True)  # type: ignore[attr-defined]
    assert gradients.shape == inputs.shape
    assert torch.isfinite(gradients).all()


def test_non_negative_linear_exposes_non_negative_weights() -> None:
    layer = NonNegativeLinear(in_features=3, out_features=4, bias=True)
    assert torch.all(layer.weight >= 0.0)
    outputs = layer(torch.randn(5, 3))
    assert outputs.shape == (5, 4)


def test_icnn_hidden_weights_are_non_negative() -> None:
    icnn = InputConvexNeuralNetwork(
        input_dim=2,
        hidden_dims=[8, 8, 8],
        activation="softplus",
        strong_convexity=0.1,
    )
    assert all(torch.all(layer.weight >= 0.0) for layer in icnn.z_layers)
    assert torch.all(icnn.output_z.weight >= 0.0)


def test_build_potential_returns_expected_types() -> None:
    mlp = build_potential(
        {
            "kind": "mlp",
            "input_dim": 2,
            "hidden_dims": [8, 8],
            "activation": "silu",
        }
    )
    icnn = build_potential(
        {
            "kind": "icnn",
            "input_dim": 2,
            "hidden_dims": [8, 8],
            "activation": "softplus",
            "strong_convexity": 0.1,
        }
    )
    assert isinstance(mlp, PotentialMLP)
    assert isinstance(icnn, InputConvexNeuralNetwork)
