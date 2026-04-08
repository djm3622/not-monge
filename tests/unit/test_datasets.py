from __future__ import annotations

import copy

import pytest
import torch

from src.datasets.celeba import DictVisionDataset, build_image_dataset_bundle
from src.datasets.diffusion_latent import DiffusionLatentDataset, build_diffusion_latent_bundle
from src.datasets.makkuva_2d import build_makkuva_2d_benchmark
from src.datasets.synthetic_ot import TensorDictDataset, build_synthetic_ot_benchmark
from src.utils.data import collect_loader_tensors, maybe_override_batch_size, split_tensor_dict

pytestmark = pytest.mark.unit


def test_tensor_dict_dataset_returns_consistent_shapes() -> None:
    dataset = TensorDictDataset(
        {
            "source": torch.randn(5, 2),
            "target": torch.randn(5, 2),
        }
    )
    sample = dataset[0]
    assert len(dataset) == 5
    assert sample["source"].shape == (2,)
    assert sample["target"].shape == (2,)


def test_synthetic_benchmark_is_reproducible(
    tiny_synthetic_ot_config: dict[str, object],
) -> None:
    first = build_synthetic_ot_benchmark(copy.deepcopy(tiny_synthetic_ot_config))
    second = build_synthetic_ot_benchmark(copy.deepcopy(tiny_synthetic_ot_config))
    for key in ["source", "target", "ground_truth_map"]:
        assert torch.allclose(first.train.tensors[key], second.train.tensors[key])
        assert torch.allclose(first.val.tensors[key], second.val.tensors[key])
        assert torch.allclose(first.test.tensors[key], second.test.tensors[key])


def test_synthetic_benchmark_dataloaders_have_expected_shapes(
    synthetic_bundle: object,
) -> None:
    train_loader, val_loader, test_loader = synthetic_bundle.make_dataloaders()
    train_batch = next(iter(train_loader))
    val_batch = next(iter(val_loader))
    test_batch = next(iter(test_loader))
    for batch in [train_batch, val_batch, test_batch]:
        assert batch["source"].shape[-1] == 2
        assert batch["target"].shape == batch["source"].shape
        assert batch["ground_truth_map"].shape == batch["source"].shape


def test_makkuva_benchmark_is_reproducible_and_unsupervised(
    tiny_makkuva_checkerboard_config: dict[str, object],
) -> None:
    first = build_makkuva_2d_benchmark(copy.deepcopy(tiny_makkuva_checkerboard_config))
    second = build_makkuva_2d_benchmark(copy.deepcopy(tiny_makkuva_checkerboard_config))
    first_train, first_val, first_test = first.make_dataloaders()
    second_train, second_val, second_test = second.make_dataloaders()

    first_train_batch = next(iter(first_train))
    second_train_batch = next(iter(second_train))
    assert torch.allclose(first_train_batch["source"], second_train_batch["source"])
    assert torch.allclose(first_train_batch["target"], second_train_batch["target"])

    first_val_batch = next(iter(first_val))
    second_val_batch = next(iter(second_val))
    first_test_batch = next(iter(first_test))
    second_test_batch = next(iter(second_test))
    assert "ground_truth_map" not in first_train_batch
    assert "ground_truth_map" not in first_val_batch
    assert torch.allclose(first_val_batch["source"], second_val_batch["source"])
    assert torch.allclose(first_test_batch["target"], second_test_batch["target"])
    assert first_val_batch["source"].shape[-1] == 2
    assert first_val_batch["target"].shape == first_val_batch["source"].shape


def test_fake_image_dataset_bundle_returns_dict_samples(
    tiny_fake_image_dataset_config: dict[str, object],
) -> None:
    bundle = build_image_dataset_bundle(copy.deepcopy(tiny_fake_image_dataset_config))
    assert isinstance(bundle.train, DictVisionDataset)
    sample = bundle.train[0]
    assert sample["image"].shape == (3, 16, 16)
    assert sample["label"].ndim == 0


def test_diffusion_latent_bundle_builds_from_fake_components(
    fake_timm: object,
    tiny_diffusion_latent_dataset_config: dict[str, object],
) -> None:
    del fake_timm
    bundle = build_diffusion_latent_bundle(copy.deepcopy(tiny_diffusion_latent_dataset_config))
    assert isinstance(bundle.train, DiffusionLatentDataset)
    sample = bundle.train[0]
    assert sample["source"].shape == (8,)
    assert sample["target"].shape == (8,)
    assert sample["clean_latent"].shape == (8,)
    assert sample["ground_truth_map"].shape == (8,)


def test_data_helpers_collect_and_split(
    synthetic_bundle: object,
) -> None:
    train_loader, _, _ = synthetic_bundle.make_dataloaders()
    collected = collect_loader_tensors(train_loader, keys=["source", "target"], max_items=10)
    assert collected["source"].shape[0] == 10
    assert collected["target"].shape == collected["source"].shape

    updated = maybe_override_batch_size({"name": "synthetic_ot", "batch_size": 8}, batch_size=4)
    assert updated["batch_size"] == 4

    split_batches = list(split_tensor_dict(collected, batch_size=4))
    assert [batch["source"].shape[0] for batch in split_batches] == [4, 4, 2]
