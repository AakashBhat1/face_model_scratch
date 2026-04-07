from __future__ import annotations

from pathlib import Path

from torch.utils.data import DataLoader
from torchvision import datasets, transforms


def build_transforms(image_size: int) -> tuple[transforms.Compose, transforms.Compose]:
    train_transform = transforms.Compose(
        [
            transforms.Resize((image_size + 16, image_size + 16)),
            transforms.RandomCrop(image_size),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ColorJitter(brightness=0.2, contrast=0.2),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ]
    )
    val_transform = transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ]
    )
    return train_transform, val_transform


def create_dataloaders(
    data_root: Path,
    image_size: int,
    batch_size: int,
    num_workers: int,
) -> tuple[DataLoader, DataLoader, int, list[str]]:
    train_dir = data_root / "train"
    val_dir = data_root / "val"
    if not train_dir.exists():
        raise FileNotFoundError(f"Missing train directory: {train_dir}")
    if not val_dir.exists():
        raise FileNotFoundError(f"Missing val directory: {val_dir}")

    train_t, val_t = build_transforms(image_size=image_size)
    train_ds = datasets.ImageFolder(root=str(train_dir), transform=train_t)
    val_ds = datasets.ImageFolder(root=str(val_dir), transform=val_t)

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=False,
    )
    return train_loader, val_loader, len(train_ds.classes), train_ds.classes
