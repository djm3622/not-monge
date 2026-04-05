from __future__ import annotations

import copy
import types

import pytest
import torch
from torch import nn

import src.models.unet_diffusion as unet_module
from src.models.common import SinusoidalTimeEmbedding, get_activation
from src.models.groupsort import GroupSort, GroupSortMLP
from src.models.ot_map import OTMapNetwork, ResidualMLPBlock, build_ot_map
from src.models.unet_diffusion import DiffusionUNet, build_diffusion_model
from src.utils.seed import seed_all

pytestmark = pytest.mark.unit


@pytest.mark.parametrize("name", ["relu", "gelu", "elu", "tanh", "softplus", "leaky_relu", "silu"])
def test_get_activation_returns_module(name: str) -> None:
    assert isinstance(get_activation(name), nn.Module)


def test_residual_mlp_block_preserves_shape() -> None:
    block = ResidualMLPBlock(hidden_dim=8, activation="silu", dropout=0.0, layer_norm=True)
    inputs = torch.randn(4, 8)
    outputs = block(inputs)
    assert outputs.shape == inputs.shape
    assert torch.isfinite(outputs).all()


def test_ot_map_forward_shape_and_no_nans(ot_model_config: dict[str, object]) -> None:
    model = OTMapNetwork(**ot_model_config)
    inputs = torch.randn(5, int(ot_model_config["input_dim"]))
    outputs = model(inputs)
    assert outputs.shape == (5, int(ot_model_config["output_dim"]))
    assert torch.isfinite(outputs).all()


def test_build_ot_map_is_deterministic(ot_model_config: dict[str, object]) -> None:
    seed_all(77, deterministic=True)
    model_a = build_ot_map(ot_model_config)
    inputs = torch.randn(4, int(ot_model_config["input_dim"]))
    outputs_a = model_a(inputs)
    seed_all(77, deterministic=True)
    model_b = build_ot_map(ot_model_config)
    outputs_b = model_b(inputs)
    assert torch.allclose(outputs_a, outputs_b, atol=1.0e-6, rtol=1.0e-5)


def test_groupsort_sorts_each_group() -> None:
    activation = GroupSort(group_size=2)
    inputs = torch.tensor([[3.0, 1.0, 4.0, 2.0]])
    outputs = activation(inputs)
    expected = torch.tensor([[1.0, 3.0, 2.0, 4.0]])
    assert torch.equal(outputs, expected)


def test_groupsort_raises_when_features_not_divisible() -> None:
    activation = GroupSort(group_size=2)
    with pytest.raises(ValueError, match="divisible"):
        activation(torch.randn(2, 3))


def test_groupsort_mlp_forward_shape() -> None:
    model = GroupSortMLP(input_dim=2, hidden_dims=[8, 8], output_dim=1, group_size=2)
    outputs = model(torch.randn(6, 2))
    assert outputs.shape == (6, 1)
    assert torch.isfinite(outputs).all()


def test_sinusoidal_time_embedding_shape_and_determinism() -> None:
    embedding = SinusoidalTimeEmbedding(embedding_dim=7)
    timesteps = torch.tensor([0, 1, 10, 100])
    first = embedding(timesteps)
    second = embedding(timesteps)
    assert first.shape == (4, 7)
    assert torch.allclose(first, second, atol=1.0e-6, rtol=1.0e-5)


def test_custom_diffusion_unet_forward_shape_and_no_nans(
    tiny_diffusion_model_config: dict[str, object],
) -> None:
    model = DiffusionUNet(
        sample_size=int(tiny_diffusion_model_config["sample_size"]),
        in_channels=int(tiny_diffusion_model_config["in_channels"]),
        out_channels=int(tiny_diffusion_model_config["out_channels"]),
        base_channels=int(tiny_diffusion_model_config["base_channels"]),
        channel_multipliers=list(tiny_diffusion_model_config["channel_multipliers"]),
        num_res_blocks=int(tiny_diffusion_model_config["num_res_blocks"]),
        attention_resolutions=list(tiny_diffusion_model_config["attention_resolutions"]),
        dropout=float(tiny_diffusion_model_config["dropout"]),
        num_heads=int(tiny_diffusion_model_config["num_heads"]),
    )
    images = torch.randn(2, 3, 16, 16)
    timesteps = torch.tensor([1, 2], dtype=torch.long)
    prediction = model(images, timesteps)
    assert prediction.shape == images.shape
    assert torch.isfinite(prediction).all()


def test_diffusers_wrapper_uses_monkeypatched_backend(
    monkeypatch: pytest.MonkeyPatch,
    tiny_diffusion_model_config: dict[str, object],
) -> None:
    class FakeUNet2DModel(nn.Module):
        def __init__(self, **kwargs: object) -> None:
            super().__init__()
            self.kwargs = kwargs

        def forward(self, x: torch.Tensor, timesteps: torch.Tensor) -> types.SimpleNamespace:
            del timesteps
            return types.SimpleNamespace(sample=torch.zeros_like(x) + 0.25)

    monkeypatch.setattr(unet_module, "UNet2DModel", FakeUNet2DModel)
    config = copy.deepcopy(tiny_diffusion_model_config)
    config["backend"] = "diffusers"
    model = build_diffusion_model(config)
    outputs = model(torch.randn(2, 3, 16, 16), torch.tensor([1, 2], dtype=torch.long))
    assert outputs.shape == (2, 3, 16, 16)
    assert torch.allclose(outputs, torch.full_like(outputs, 0.25))


def test_swin_diffusion_model_with_fake_timm(
    fake_timm: object,
) -> None:
    del fake_timm
    config = {
        "name": "diffusion_swin",
        "in_channels": 3,
        "out_channels": 3,
        "backbone_name": "fake_swin",
        "pretrained_backbone": False,
        "decoder_channels": 32,
    }
    model = build_diffusion_model(config)
    images = torch.randn(2, 3, 16, 16)
    timesteps = torch.tensor([0, 5], dtype=torch.long)
    outputs = model(images, timesteps)
    assert outputs.shape == images.shape
    assert torch.isfinite(outputs).all()
