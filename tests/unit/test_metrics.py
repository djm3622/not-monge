from __future__ import annotations

import pytest
import torch

from src.evaluation.concavity_metrics import (
    c_concave_envelope,
    convexity_violation,
    envelope_gap,
    hessian_spectrum,
    numerical_c_transform,
)
from src.evaluation.generative_metrics import (
    extract_inception_features,
    frechet_inception_distance,
    precision_recall_from_features,
)
from src.evaluation.ot_metrics import (
    empirical_kr_distance,
    empirical_w2_distance,
    gradient_error,
    map_l2_error,
    maximum_mean_discrepancy,
)

pytestmark = pytest.mark.unit


def test_ot_metrics_are_zero_or_near_zero_for_matching_samples() -> None:
    samples = torch.randn(16, 2)
    assert torch.isclose(map_l2_error(samples, samples), torch.tensor(0.0))
    assert empirical_w2_distance(samples, samples) == pytest.approx(0.0, abs=1.0e-6)
    assert empirical_kr_distance(samples, samples) == pytest.approx(0.0, abs=1.0e-6)
    assert maximum_mean_discrepancy(samples, samples) == pytest.approx(0.0, abs=1.0e-6)


def test_ot_metrics_are_positive_for_shifted_samples() -> None:
    samples = torch.randn(16, 2)
    shifted = samples + 2.0
    assert empirical_w2_distance(samples, shifted) > 0.1
    assert empirical_kr_distance(samples, shifted) > 0.1
    assert maximum_mean_discrepancy(samples, shifted) > 0.0


def test_gradient_error_is_zero_for_identical_maps() -> None:
    inputs = torch.randn(8, 2)

    def same_map(x: torch.Tensor) -> torch.Tensor:
        return 2.0 * x

    error = gradient_error(same_map, same_map, inputs)
    assert error == pytest.approx(0.0, abs=1.0e-6)


def test_c_concavity_helpers_on_quadratic_potential() -> None:
    support = torch.randn(12, 2)
    values = 0.5 * support.pow(2).sum(dim=-1, keepdim=True)
    transformed = numerical_c_transform(support, support, values)
    envelope = c_concave_envelope(support, values)
    gap_metrics = envelope_gap(support, values)
    assert transformed.shape == (support.shape[0],)
    assert envelope.shape == (support.shape[0],)
    assert gap_metrics["envelope_gap/mean"] >= 0.0
    assert gap_metrics["envelope_gap/max"] >= 0.0


def test_convexity_and_hessian_metrics_for_quadratic_potential() -> None:
    def quadratic(x: torch.Tensor) -> torch.Tensor:
        return 0.5 * x.pow(2).sum(dim=-1, keepdim=True)

    points = torch.randn(16, 2)
    convexity = convexity_violation(quadratic, points)
    spectrum = hessian_spectrum(quadratic, points)
    assert convexity["convexity_violation/mean"] == pytest.approx(0.0, abs=1.0e-5)
    assert convexity["convexity_violation/max"] == pytest.approx(0.0, abs=1.0e-5)
    assert spectrum["hessian/min_eig"] == pytest.approx(1.0, abs=1.0e-4)
    assert spectrum["hessian/max_eig"] == pytest.approx(1.0, abs=1.0e-4)


def test_fid_and_precision_recall_behave_sensibly() -> None:
    real = torch.randn(16, 8)
    fake = real.clone()
    shifted = real + 5.0
    assert frechet_inception_distance(real, fake) == pytest.approx(0.0, abs=1.0e-6)
    identical = precision_recall_from_features(real, fake)
    separated = precision_recall_from_features(real, shifted)
    assert identical["precision"] == pytest.approx(1.0, abs=1.0e-6)
    assert identical["recall"] == pytest.approx(1.0, abs=1.0e-6)
    assert separated["precision"] <= identical["precision"]
    assert separated["recall"] <= identical["recall"]


def test_extract_inception_features_uses_fake_extractor(
    fake_inception_extractor: object,
) -> None:
    del fake_inception_extractor
    images = torch.randn(6, 3, 16, 16)
    features = extract_inception_features(images, device=torch.device("cpu"), batch_size=2)
    assert features.shape == (6, 8)
    assert torch.isfinite(features).all()
