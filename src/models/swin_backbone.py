"""Swin-based encoder-decoder building blocks."""

from __future__ import annotations

from typing import Sequence

import torch
import torch.nn.functional as F
from torch import nn

from src.models.common import SinusoidalTimeEmbedding

try:
    import timm
except ImportError:  # pragma: no cover - optional dependency resolved at install time
    timm = None


class SwinFeatureBackbone(nn.Module):
    """Multi-scale feature extractor backed by timm Swin models."""

    def __init__(
        self,
        model_name: str,
        pretrained: bool = False,
        out_indices: Sequence[int] = (0, 1, 2, 3),
    ) -> None:
        super().__init__()
        if timm is None:
            raise ImportError("timm is required for Swin backbone experiments")
        self.backbone = timm.create_model(
            model_name,
            pretrained=pretrained,
            features_only=True,
            out_indices=tuple(out_indices),
        )
        self.feature_channels = list(self.backbone.feature_info.channels())

    def forward(self, images: torch.Tensor) -> list[torch.Tensor]:
        if images.shape[-1] != 224 or images.shape[-2] != 224:
            images = F.interpolate(images, size=(224, 224), mode="bilinear", align_corners=False)
        return list(self.backbone(images))


class DecoderBlock(nn.Module):
    """Simple upsampling decoder block."""

    def __init__(self, in_channels: int, skip_channels: int, out_channels: int) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels + skip_channels, out_channels, kernel_size=3, padding=1),
            nn.GroupNorm(num_groups=8, num_channels=out_channels),
            nn.SiLU(),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.GroupNorm(num_groups=8, num_channels=out_channels),
            nn.SiLU(),
        )

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)
        return self.block(torch.cat([x, skip], dim=1))


class SwinUNetDiffusionModel(nn.Module):
    """Swin encoder paired with a lightweight convolutional decoder."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        backbone_name: str = "swin_tiny_patch4_window7_224",
        pretrained_backbone: bool = False,
        decoder_channels: int = 128,
    ) -> None:
        super().__init__()
        self.backbone = SwinFeatureBackbone(
            model_name=backbone_name,
            pretrained=pretrained_backbone,
        )
        feature_channels = self.backbone.feature_channels
        time_dim = decoder_channels * 4
        self.time_embedding = nn.Sequential(
            SinusoidalTimeEmbedding(time_dim),
            nn.Linear(time_dim, time_dim),
            nn.SiLU(),
            nn.Linear(time_dim, feature_channels[-1]),
        )
        self.bottleneck = nn.Sequential(
            nn.Conv2d(feature_channels[-1], decoder_channels, kernel_size=3, padding=1),
            nn.GroupNorm(num_groups=8, num_channels=decoder_channels),
            nn.SiLU(),
        )
        self.decoders = nn.ModuleList(
            [
                DecoderBlock(decoder_channels, feature_channels[-2], decoder_channels),
                DecoderBlock(decoder_channels, feature_channels[-3], decoder_channels // 2),
                DecoderBlock(decoder_channels // 2, feature_channels[-4], decoder_channels // 4),
            ]
        )
        self.output = nn.Sequential(
            nn.Conv2d(decoder_channels // 4, decoder_channels // 4, kernel_size=3, padding=1),
            nn.SiLU(),
            nn.Conv2d(decoder_channels // 4, out_channels, kernel_size=3, padding=1),
        )

    def forward(self, x: torch.Tensor, timesteps: torch.Tensor) -> torch.Tensor:
        original_size = x.shape[-2:]
        features = self.backbone(x)
        bottleneck = features[-1] + self.time_embedding(timesteps).unsqueeze(-1).unsqueeze(-1)
        hidden = self.bottleneck(bottleneck)
        for decoder, skip in zip(self.decoders, reversed(features[:-1])):
            hidden = decoder(hidden, skip)
        hidden = F.interpolate(hidden, size=original_size, mode="bilinear", align_corners=False)
        return self.output(hidden)
