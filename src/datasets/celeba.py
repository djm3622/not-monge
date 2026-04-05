"""Torchvision image dataset wrappers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Iterable

import torch
from torch.utils.data import DataLoader, Dataset, random_split
from torchvision import datasets, transforms


class DictVisionDataset(Dataset[dict[str, torch.Tensor]]):
    """Wrap torchvision datasets to return tensor dictionaries."""

    def __init__(self, dataset: Dataset[Any]) -> None:
        self.dataset = dataset

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        item = self.dataset[index]
        if isinstance(item, tuple):
            image = item[0]
            label = item[1] if len(item) > 1 else -1
        else:
            image = item
            label = -1
        label_tensor = torch.as_tensor(label) if not torch.is_tensor(label) else label
        return {"image": image, "label": label_tensor}


@dataclass
class ImageDatasetBundle:
    """Train/validation/test splits for image experiments."""

    train: Dataset[dict[str, torch.Tensor]]
    val: Dataset[dict[str, torch.Tensor]]
    test: Dataset[dict[str, torch.Tensor]]
    batch_size: int
    num_workers: int

    def make_dataloaders(self) -> tuple[DataLoader[dict[str, torch.Tensor]], ...]:
        loader_kwargs = {
            "batch_size": self.batch_size,
            "num_workers": self.num_workers,
            "pin_memory": False,
        }
        train_loader = DataLoader(self.train, shuffle=True, drop_last=True, **loader_kwargs)
        val_loader = DataLoader(self.val, shuffle=False, drop_last=False, **loader_kwargs)
        test_loader = DataLoader(self.test, shuffle=False, drop_last=False, **loader_kwargs)
        return train_loader, val_loader, test_loader


def image_transform(image_size: int) -> Callable[[Any], torch.Tensor]:
    """Standard image transform with normalization to [-1, 1]."""
    return transforms.Compose(
        [
            transforms.Resize(image_size),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ]
    )


def build_image_dataset_bundle(config: dict[str, Any]) -> ImageDatasetBundle:
    """Create either a CIFAR-10 or CelebA dataset bundle."""
    dataset_name = str(config["name"]).lower()
    root = str(config.get("root", "data"))
    transform = image_transform(int(config["image_size"]))
    batch_size = int(config.get("batch_size", 64))
    num_workers = int(config.get("num_workers", 0))
    download = bool(config.get("download", True))

    if dataset_name == "celeba":
        train_dataset = datasets.CelebA(
            root=root,
            split="train",
            target_type="attr",
            transform=transform,
            download=download,
        )
        val_dataset = datasets.CelebA(
            root=root,
            split="valid",
            target_type="attr",
            transform=transform,
            download=download,
        )
        test_dataset = datasets.CelebA(
            root=root,
            split="test",
            target_type="attr",
            transform=transform,
            download=download,
        )
    elif dataset_name == "cifar10":
        full_train = datasets.CIFAR10(root=root, train=True, transform=transform, download=download)
        train_length = int(0.9 * len(full_train))
        val_length = len(full_train) - train_length
        generator = torch.Generator().manual_seed(1234)
        train_dataset, val_dataset = random_split(full_train, [train_length, val_length], generator=generator)
        test_dataset = datasets.CIFAR10(root=root, train=False, transform=transform, download=download)
    elif dataset_name == "fake_data":
        image_shape = (3, int(config["image_size"]), int(config["image_size"]))
        train_dataset = datasets.FakeData(
            size=int(config.get("train_size", 128)),
            image_size=image_shape,
            num_classes=10,
            transform=transform,
        )
        val_dataset = datasets.FakeData(
            size=int(config.get("val_size", 32)),
            image_size=image_shape,
            num_classes=10,
            transform=transform,
        )
        test_dataset = datasets.FakeData(
            size=int(config.get("test_size", 32)),
            image_size=image_shape,
            num_classes=10,
            transform=transform,
        )
    else:
        raise ValueError(f"Unsupported image dataset: {dataset_name}")

    return ImageDatasetBundle(
        train=DictVisionDataset(train_dataset),
        val=DictVisionDataset(val_dataset),
        test=DictVisionDataset(test_dataset),
        batch_size=batch_size,
        num_workers=num_workers,
    )


def denormalize_images(images: torch.Tensor) -> torch.Tensor:
    """Map images from [-1, 1] back to [0, 1]."""
    return images.mul(0.5).add(0.5).clamp(0.0, 1.0)
