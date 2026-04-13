from __future__ import annotations

from pathlib import Path

from torch.utils.data import Subset
from torchvision import datasets, transforms

from .models import dataset_name_for_model


def get_datasets(model_name: str, data_dir: str | Path):
    data_dir = Path(data_dir)
    transform = transforms.ToTensor()
    dataset_name = dataset_name_for_model(model_name)

    if dataset_name == "mnist":
        train_ds = datasets.MNIST(root=data_dir, train=True, download=True, transform=transform)
        test_ds = datasets.MNIST(root=data_dir, train=False, download=True, transform=transform)
        return train_ds, test_ds

    if dataset_name == "cifar10":
        train_ds = datasets.CIFAR10(root=data_dir, train=True, download=True, transform=transform)
        test_ds = datasets.CIFAR10(root=data_dir, train=False, download=True, transform=transform)
        return train_ds, test_ds

    raise ValueError(f"Unsupported dataset for model {model_name}")


def maybe_subset(dataset, subset_size: int | None):
    if subset_size is None or subset_size <= 0 or subset_size >= len(dataset):
        return dataset
    return Subset(dataset, list(range(subset_size)))
