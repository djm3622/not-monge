"""Generative-model evaluation metrics."""

from __future__ import annotations

from typing import Iterable

import numpy as np
import torch
import torch.nn.functional as F
from scipy import linalg
from torch import nn
from torchvision.models import Inception_V3_Weights, inception_v3

from src.datasets.celeba import denormalize_images


class InceptionFeatureExtractor(nn.Module):
    """Inception-v3 pooled features for FID and precision/recall."""

    def __init__(self) -> None:
        super().__init__()
        weights = Inception_V3_Weights.IMAGENET1K_V1
        model = inception_v3(weights=weights, transform_input=False, aux_logits=False)
        model.fc = nn.Identity()
        model.eval()
        self.model = model
        self.mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
        self.std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        images = denormalize_images(images)
        images = F.interpolate(images, size=(299, 299), mode="bilinear", align_corners=False)
        mean = self.mean.to(images.device, images.dtype)
        std = self.std.to(images.device, images.dtype)
        images = (images - mean) / std
        return self.model(images)


@torch.no_grad()
def extract_inception_features(
    images: torch.Tensor,
    device: torch.device,
    batch_size: int = 64,
) -> torch.Tensor:
    """Extract pooled Inception features."""
    extractor = InceptionFeatureExtractor().to(device)
    features: list[torch.Tensor] = []
    for chunk in images.split(batch_size):
        features.append(extractor(chunk.to(device)).cpu())
    return torch.cat(features, dim=0)


def frechet_inception_distance(real_features: torch.Tensor, fake_features: torch.Tensor) -> float:
    """Compute FID from feature arrays."""
    real = real_features.numpy()
    fake = fake_features.numpy()
    mu_real = real.mean(axis=0)
    mu_fake = fake.mean(axis=0)
    sigma_real = np.cov(real, rowvar=False)
    sigma_fake = np.cov(fake, rowvar=False)
    diff = mu_real - mu_fake
    covmean, _ = linalg.sqrtm(sigma_real @ sigma_fake, disp=False)
    if np.iscomplexobj(covmean):
        covmean = covmean.real
    fid = diff.dot(diff) + np.trace(sigma_real + sigma_fake - 2.0 * covmean)
    return float(np.real(fid))


def precision_recall_from_features(
    real_features: torch.Tensor,
    fake_features: torch.Tensor,
    nearest_k: int = 3,
) -> dict[str, float]:
    """Compute manifold precision/recall from feature embeddings."""
    real_distances = torch.cdist(real_features, real_features)
    fake_distances = torch.cdist(fake_features, fake_features)
    real_radius = real_distances.topk(nearest_k + 1, largest=False).values[:, -1]
    fake_radius = fake_distances.topk(nearest_k + 1, largest=False).values[:, -1]
    cross_distances = torch.cdist(fake_features, real_features)
    precision = (cross_distances <= real_radius.unsqueeze(0)).any(dim=1).float().mean()
    recall = (cross_distances.transpose(0, 1) <= fake_radius.unsqueeze(0)).any(dim=1).float().mean()
    return {"precision": float(precision), "recall": float(recall)}
