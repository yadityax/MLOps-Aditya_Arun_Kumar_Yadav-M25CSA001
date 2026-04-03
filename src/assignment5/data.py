from __future__ import annotations

from dataclasses import dataclass

import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader


def _loader_kwargs(num_workers: int) -> dict[str, object]:
    pin_memory = torch.cuda.is_available()
    worker_count = num_workers if pin_memory else 0
    kwargs: dict[str, object] = {
        "num_workers": worker_count,
        "pin_memory": pin_memory,
    }
    if worker_count > 0:
        kwargs["persistent_workers"] = True
    return kwargs


@dataclass(slots=True)
class DataBundle:
    train_loader: DataLoader
    val_loader: DataLoader
    test_loader: DataLoader
    num_classes: int


def build_cifar100_loaders(data_root: str, batch_size: int, num_workers: int, image_size: int) -> DataBundle:
    normalize = transforms.Normalize(
        mean=(0.5071, 0.4867, 0.4408),
        std=(0.2675, 0.2565, 0.2761),
    )

    train_transform = transforms.Compose(
        [
            transforms.RandomResizedCrop(image_size, scale=(0.8, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ]
    )

    eval_transform = transforms.Compose(
        [
            transforms.Resize(image_size + 32),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
            normalize,
        ]
    )

    train_dataset = datasets.CIFAR100(root=data_root, train=True, download=True, transform=train_transform)
    test_dataset = datasets.CIFAR100(root=data_root, train=False, download=True, transform=eval_transform)

    val_size = 5000
    train_size = len(train_dataset) - val_size
    train_dataset, val_dataset = torch.utils.data.random_split(
        train_dataset,
        [train_size, val_size],
        generator=torch.Generator().manual_seed(42),
    )

    loader_kwargs = _loader_kwargs(num_workers)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, **loader_kwargs)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, **loader_kwargs)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, **loader_kwargs)
    return DataBundle(train_loader=train_loader, val_loader=val_loader, test_loader=test_loader, num_classes=100)


def build_cifar10_loaders(data_root: str, batch_size: int, num_workers: int, image_size: int) -> DataBundle:
    normalize = transforms.Normalize(
        mean=(0.4914, 0.4822, 0.4465),
        std=(0.2470, 0.2435, 0.2616),
    )

    train_transform = transforms.Compose(
        [
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.Resize(image_size),
            transforms.ToTensor(),
            normalize,
        ]
    )

    eval_transform = transforms.Compose(
        [
            transforms.Resize(image_size),
            transforms.ToTensor(),
            normalize,
        ]
    )

    train_dataset = datasets.CIFAR10(root=data_root, train=True, download=True, transform=train_transform)
    test_dataset = datasets.CIFAR10(root=data_root, train=False, download=True, transform=eval_transform)

    val_size = 5000
    train_size = len(train_dataset) - val_size
    train_dataset, val_dataset = torch.utils.data.random_split(
        train_dataset,
        [train_size, val_size],
        generator=torch.Generator().manual_seed(42),
    )

    loader_kwargs = _loader_kwargs(num_workers)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, **loader_kwargs)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, **loader_kwargs)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, **loader_kwargs)
    return DataBundle(train_loader=train_loader, val_loader=val_loader, test_loader=test_loader, num_classes=10)
