"""Diffusion backbones with custom and diffusers-based U-Nets."""

from __future__ import annotations

from typing import Any, Mapping, Sequence

import torch
import torch.nn.functional as F
from torch import nn

from src.models.common import SinusoidalTimeEmbedding
from src.models.swin_backbone import SwinUNetDiffusionModel

try:
    from diffusers import UNet2DModel
except ImportError:  # pragma: no cover - optional dependency resolved at install time
    UNet2DModel = None


class ResidualBlock2D(nn.Module):
    """Residual convolution block with timestep conditioning."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        time_dim: int,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.norm1 = nn.GroupNorm(num_groups=8, num_channels=in_channels)
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.time_projection = nn.Linear(time_dim, out_channels)
        self.norm2 = nn.GroupNorm(num_groups=8, num_channels=out_channels)
        self.dropout = nn.Dropout(dropout)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.skip = (
            nn.Conv2d(in_channels, out_channels, kernel_size=1)
            if in_channels != out_channels
            else nn.Identity()
        )

    def forward(self, x: torch.Tensor, time_embedding: torch.Tensor) -> torch.Tensor:
        hidden = self.conv1(F.silu(self.norm1(x)))
        hidden = hidden + self.time_projection(time_embedding).unsqueeze(-1).unsqueeze(-1)
        hidden = self.conv2(self.dropout(F.silu(self.norm2(hidden))))
        return hidden + self.skip(x)


class AttentionBlock2D(nn.Module):
    """Spatial self-attention using PyTorch SDPA."""

    def __init__(self, channels: int, num_heads: int = 4) -> None:
        super().__init__()
        if channels % num_heads != 0:
            raise ValueError("channels must be divisible by num_heads")
        self.channels = channels
        self.num_heads = num_heads
        self.head_dim = channels // num_heads
        self.norm = nn.GroupNorm(num_groups=8, num_channels=channels)
        self.to_qkv = nn.Conv2d(channels, channels * 3, kernel_size=1)
        self.proj = nn.Conv2d(channels, channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        batch_size, channels, height, width = x.shape
        qkv = self.to_qkv(self.norm(x))
        q, k, v = qkv.chunk(3, dim=1)
        q = q.view(batch_size, self.num_heads, self.head_dim, height * width).transpose(-2, -1)
        k = k.view(batch_size, self.num_heads, self.head_dim, height * width).transpose(-2, -1)
        v = v.view(batch_size, self.num_heads, self.head_dim, height * width).transpose(-2, -1)
        attended = F.scaled_dot_product_attention(q, k, v)
        attended = attended.transpose(-2, -1).reshape(batch_size, channels, height, width)
        return residual + self.proj(attended)


class Downsample2D(nn.Module):
    """Strided convolution downsampling."""

    def __init__(self, channels: int) -> None:
        super().__init__()
        self.conv = nn.Conv2d(channels, channels, kernel_size=3, stride=2, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class Upsample2D(nn.Module):
    """Nearest-neighbor upsampling followed by convolution."""

    def __init__(self, channels: int) -> None:
        super().__init__()
        self.conv = nn.Conv2d(channels, channels, kernel_size=3, padding=1)

    def forward(self, x: torch.Tensor, size: tuple[int, int]) -> torch.Tensor:
        x = F.interpolate(x, size=size, mode="nearest")
        return self.conv(x)


class EncoderStage(nn.Module):
    """Encoder stage with residual blocks and optional attention."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        time_dim: int,
        num_res_blocks: int,
        use_attention: bool,
        dropout: float,
        num_heads: int,
        downsample: bool,
    ) -> None:
        super().__init__()
        blocks: list[nn.Module] = []
        current_channels = in_channels
        for _ in range(num_res_blocks):
            blocks.append(ResidualBlock2D(current_channels, out_channels, time_dim, dropout))
            if use_attention:
                blocks.append(AttentionBlock2D(out_channels, num_heads))
            current_channels = out_channels
        self.blocks = nn.ModuleList(blocks)
        self.downsample = Downsample2D(out_channels) if downsample else nn.Identity()

    def forward(self, x: torch.Tensor, time_embedding: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        for block in self.blocks:
            if isinstance(block, ResidualBlock2D):
                x = block(x, time_embedding)
            else:
                x = block(x)
        skip = x
        return self.downsample(x), skip


class DecoderStage(nn.Module):
    """Decoder stage with skip connections."""

    def __init__(
        self,
        in_channels: int,
        skip_channels: int,
        out_channels: int,
        time_dim: int,
        num_res_blocks: int,
        use_attention: bool,
        dropout: float,
        num_heads: int,
    ) -> None:
        super().__init__()
        self.upsample = Upsample2D(in_channels)
        blocks: list[nn.Module] = []
        current_channels = in_channels + skip_channels
        for _ in range(num_res_blocks):
            blocks.append(ResidualBlock2D(current_channels, out_channels, time_dim, dropout))
            if use_attention:
                blocks.append(AttentionBlock2D(out_channels, num_heads))
            current_channels = out_channels
        self.blocks = nn.ModuleList(blocks)

    def forward(
        self,
        x: torch.Tensor,
        skip: torch.Tensor,
        time_embedding: torch.Tensor,
    ) -> torch.Tensor:
        x = self.upsample(x, size=skip.shape[-2:])
        x = torch.cat([x, skip], dim=1)
        for block in self.blocks:
            if isinstance(block, ResidualBlock2D):
                x = block(x, time_embedding)
            else:
                x = block(x)
        return x


class DiffusionUNet(nn.Module):
    """Compact DDPM U-Net for 32x32 or 64x64 image generation."""

    def __init__(
        self,
        sample_size: int,
        in_channels: int,
        out_channels: int,
        base_channels: int,
        channel_multipliers: Sequence[int],
        num_res_blocks: int,
        attention_resolutions: Sequence[int],
        dropout: float,
        num_heads: int,
    ) -> None:
        super().__init__()
        self.sample_size = sample_size
        stage_channels = [base_channels * multiplier for multiplier in channel_multipliers]
        time_dim = base_channels * 4
        self.time_embedding = nn.Sequential(
            SinusoidalTimeEmbedding(time_dim),
            nn.Linear(time_dim, time_dim),
            nn.SiLU(),
            nn.Linear(time_dim, time_dim),
        )
        self.input_projection = nn.Conv2d(in_channels, stage_channels[0], kernel_size=3, padding=1)
        self.encoder = nn.ModuleList()
        current_channels = stage_channels[0]
        current_resolution = sample_size
        for index, out_stage_channels in enumerate(stage_channels):
            self.encoder.append(
                EncoderStage(
                    in_channels=current_channels,
                    out_channels=out_stage_channels,
                    time_dim=time_dim,
                    num_res_blocks=num_res_blocks,
                    use_attention=current_resolution in attention_resolutions,
                    dropout=dropout,
                    num_heads=num_heads,
                    downsample=index < len(stage_channels) - 1,
                )
            )
            current_channels = out_stage_channels
            if index < len(stage_channels) - 1:
                current_resolution //= 2
        self.mid_block = nn.Sequential(
            ResidualBlock2D(current_channels, current_channels, time_dim, dropout),
            AttentionBlock2D(current_channels, num_heads),
            ResidualBlock2D(current_channels, current_channels, time_dim, dropout),
        )
        self.decoder = nn.ModuleList()
        reversed_channels = list(reversed(stage_channels[:-1]))
        for skip_channels in reversed_channels:
            current_resolution *= 2
            self.decoder.append(
                DecoderStage(
                    in_channels=current_channels,
                    skip_channels=skip_channels,
                    out_channels=skip_channels,
                    time_dim=time_dim,
                    num_res_blocks=num_res_blocks,
                    use_attention=current_resolution in attention_resolutions,
                    dropout=dropout,
                    num_heads=num_heads,
                )
            )
            current_channels = skip_channels
        self.output = nn.Sequential(
            nn.GroupNorm(num_groups=8, num_channels=current_channels),
            nn.SiLU(),
            nn.Conv2d(current_channels, out_channels, kernel_size=3, padding=1),
        )

    def forward(self, x: torch.Tensor, timesteps: torch.Tensor) -> torch.Tensor:
        time_embedding = self.time_embedding(timesteps)
        hidden = self.input_projection(x)
        skips: list[torch.Tensor] = []
        for stage in self.encoder:
            hidden, skip = stage(hidden, time_embedding)
            skips.append(skip)
        mid_hidden = skips.pop()
        hidden = mid_hidden
        for layer in self.mid_block:
            if isinstance(layer, ResidualBlock2D):
                hidden = layer(hidden, time_embedding)
            else:
                hidden = layer(hidden)
        for stage, skip in zip(self.decoder, reversed(skips)):
            hidden = stage(hidden, skip, time_embedding)
        return self.output(hidden)


class DiffusersUNetWrapper(nn.Module):
    """Thin wrapper around diffusers.UNet2DModel."""

    def __init__(self, config: Mapping[str, Any]) -> None:
        super().__init__()
        if UNet2DModel is None:
            raise ImportError("diffusers is required for the diffusers U-Net backend")
        channels = tuple(int(config["base_channels"]) * m for m in config["channel_multipliers"])
        block_count = len(channels)
        down_types = []
        up_types = []
        current_resolution = int(config["sample_size"])
        attention_resolutions = {int(value) for value in config["attention_resolutions"]}
        for index in range(block_count):
            down_types.append("AttnDownBlock2D" if current_resolution in attention_resolutions else "DownBlock2D")
            if index < block_count - 1:
                current_resolution //= 2
        for index in range(block_count):
            current_resolution *= 2 if index > 0 else 1
            up_types.append("AttnUpBlock2D" if current_resolution in attention_resolutions else "UpBlock2D")
        self.model = UNet2DModel(
            sample_size=int(config["sample_size"]),
            in_channels=int(config["in_channels"]),
            out_channels=int(config["out_channels"]),
            layers_per_block=int(config["num_res_blocks"]),
            block_out_channels=channels,
            down_block_types=tuple(down_types),
            up_block_types=tuple(reversed(up_types)),
        )

    def forward(self, x: torch.Tensor, timesteps: torch.Tensor) -> torch.Tensor:
        return self.model(x, timesteps).sample


def build_diffusion_model(config: Mapping[str, Any]) -> nn.Module:
    """Construct the configured diffusion model."""
    cfg = dict(config)
    name = str(cfg["name"]).lower()
    backend = str(cfg.get("backend", "custom")).lower()
    if "swin" in name:
        return SwinUNetDiffusionModel(
            in_channels=int(cfg["in_channels"]),
            out_channels=int(cfg["out_channels"]),
            backbone_name=str(cfg.get("backbone_name", "swin_tiny_patch4_window7_224")),
            pretrained_backbone=bool(cfg.get("pretrained_backbone", False)),
            decoder_channels=int(cfg.get("decoder_channels", 128)),
        )
    if backend == "diffusers":
        return DiffusersUNetWrapper(cfg)
    return DiffusionUNet(
        sample_size=int(cfg["sample_size"]),
        in_channels=int(cfg["in_channels"]),
        out_channels=int(cfg["out_channels"]),
        base_channels=int(cfg["base_channels"]),
        channel_multipliers=list(cfg["channel_multipliers"]),
        num_res_blocks=int(cfg["num_res_blocks"]),
        attention_resolutions=list(cfg.get("attention_resolutions", [])),
        dropout=float(cfg.get("dropout", 0.0)),
        num_heads=int(cfg.get("num_heads", 4)),
    )
