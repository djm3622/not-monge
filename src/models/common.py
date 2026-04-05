"""Shared model building blocks."""

from __future__ import annotations

import math

import torch
from torch import nn


def get_activation(name: str) -> nn.Module:
    """Factory for activation modules."""
    name = name.lower()
    if name == "relu":
        return nn.ReLU()
    if name == "gelu":
        return nn.GELU()
    if name == "elu":
        return nn.ELU()
    if name == "tanh":
        return nn.Tanh()
    if name == "softplus":
        return nn.Softplus()
    if name == "leaky_relu":
        return nn.LeakyReLU(0.2)
    return nn.SiLU()


class SinusoidalTimeEmbedding(nn.Module):
    """Standard sinusoidal timestep embeddings for diffusion models."""

    def __init__(self, embedding_dim: int) -> None:
        super().__init__()
        self.embedding_dim = embedding_dim

    def forward(self, timesteps: torch.Tensor) -> torch.Tensor:
        half_dim = self.embedding_dim // 2
        exponent = -math.log(10000.0) * torch.arange(
            half_dim,
            device=timesteps.device,
            dtype=torch.float32,
        ) / max(half_dim - 1, 1)
        emb = timesteps.float().unsqueeze(-1) * torch.exp(exponent).unsqueeze(0)
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=-1)
        if self.embedding_dim % 2 == 1:
            emb = torch.nn.functional.pad(emb, (0, 1))
        return emb
