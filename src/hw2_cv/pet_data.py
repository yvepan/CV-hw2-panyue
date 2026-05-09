from __future__ import annotations

import random
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import OxfordIIITPet
from torchvision.transforms import InterpolationMode


IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def classification_loaders(
    root: str | Path,
    image_size: int,
    batch_size: int,
    num_workers: int,
) -> tuple[DataLoader, DataLoader]:
    train_tf = transforms.Compose(
        [
            transforms.Resize((image_size + 32, image_size + 32)),
            transforms.RandomResizedCrop(image_size, scale=(0.7, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(0.2, 0.2, 0.2, 0.05),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]
    )
    eval_tf = transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]
    )
    train_ds = OxfordIIITPet(root=str(root), split="trainval", target_types="category", transform=train_tf, download=True)
    val_ds = OxfordIIITPet(root=str(root), split="test", target_types="category", transform=eval_tf, download=True)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)
    return train_loader, val_loader


class PetSegmentationDataset(torch.utils.data.Dataset):
    def __init__(self, root: str | Path, split: str, image_size: int, training: bool) -> None:
        self.ds = OxfordIIITPet(root=str(root), split=split, target_types="segmentation", download=True)
        self.image_size = image_size
        self.training = training
        self.image_norm = transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD)

    def __len__(self) -> int:
        return len(self.ds)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        image, mask = self.ds[idx]
        image = image.convert("RGB")
        mask = mask.convert("L")

        image = transforms.functional.resize(
            image, [self.image_size, self.image_size], interpolation=InterpolationMode.BILINEAR
        )
        mask = transforms.functional.resize(
            mask, [self.image_size, self.image_size], interpolation=InterpolationMode.NEAREST
        )
        if self.training and random.random() < 0.5:
            image = transforms.functional.hflip(image)
            mask = transforms.functional.hflip(mask)
        if self.training:
            image = transforms.ColorJitter(0.15, 0.15, 0.15, 0.03)(image)

        image_t = transforms.functional.to_tensor(image)
        image_t = self.image_norm(image_t)
        mask_np = np.asarray(mask, dtype=np.int64) - 1
        mask_np = np.clip(mask_np, 0, 2)
        mask_t = torch.from_numpy(mask_np).long()
        return image_t, mask_t


def segmentation_loaders(
    root: str | Path,
    image_size: int,
    batch_size: int,
    num_workers: int,
) -> tuple[DataLoader, DataLoader]:
    train_ds = PetSegmentationDataset(root, split="trainval", image_size=image_size, training=True)
    val_ds = PetSegmentationDataset(root, split="test", image_size=image_size, training=False)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)
    return train_loader, val_loader


def colorize_mask(mask: torch.Tensor) -> Image.Image:
    colors = np.array([[0, 0, 0], [80, 180, 80], [230, 210, 60]], dtype=np.uint8)
    arr = colors[mask.detach().cpu().numpy().clip(0, 2)]
    return Image.fromarray(arr)

