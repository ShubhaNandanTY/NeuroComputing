"""MNIST loaders.

Copied verbatim from snn-surrogate-baseline/neuron_baseline/data.py at commit
b688411 so every baseline shares the same train/val/test split. Keep
`val_size` and `seed` identical to that sub-project's Config (5000, 20260923)
or results stop being comparable.
"""
from __future__ import annotations

from pathlib import Path

import torch
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms


TUTORIAL_TRANSFORM = transforms.Compose([
    transforms.Resize((28, 28)),
    transforms.Grayscale(),
    transforms.ToTensor(),
    transforms.Normalize((0,), (1,)),
])


def _split_indices(n_total: int, val_size: int, seed: int) -> tuple[list[int], list[int]]:
    g = torch.Generator().manual_seed(seed)
    perm = torch.randperm(n_total, generator=g).tolist()
    return perm[val_size:], perm[:val_size]


def build_loaders(
    data_dir: str | Path,
    batch_size: int,
    eval_batch_size: int,
    val_size: int,
    seed: int,
    num_workers: int = 0,
    pin_memory: bool = True,
    download: bool = True,
) -> dict[str, DataLoader]:
    data_dir = Path(data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)

    full_train = datasets.MNIST(str(data_dir), train=True,  download=download, transform=TUTORIAL_TRANSFORM)
    test_ds    = datasets.MNIST(str(data_dir), train=False, download=download, transform=TUTORIAL_TRANSFORM)

    train_idx, val_idx = _split_indices(len(full_train), val_size, seed)
    train_ds = Subset(full_train, train_idx)
    val_ds   = Subset(full_train, val_idx)

    # DataLoader RNG separated from the global seed so a shifted global seed
    # doesn't silently reshuffle the val split.
    train_gen = torch.Generator().manual_seed(seed + 1)

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True, drop_last=True,
        num_workers=num_workers, pin_memory=pin_memory, generator=train_gen,
    )
    val_loader = DataLoader(
        val_ds, batch_size=eval_batch_size, shuffle=False, drop_last=False,
        num_workers=num_workers, pin_memory=pin_memory,
    )
    test_loader = DataLoader(
        test_ds, batch_size=eval_batch_size, shuffle=False, drop_last=False,
        num_workers=num_workers, pin_memory=pin_memory,
    )
    return {"train": train_loader, "val": val_loader, "test": test_loader}
