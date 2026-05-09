from __future__ import annotations

import torch
from torch import nn
import torch.nn.functional as F


class DiceLoss(nn.Module):
    def __init__(self, num_classes: int = 3, smooth: float = 1.0) -> None:
        super().__init__()
        self.num_classes = num_classes
        self.smooth = smooth

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        probs = torch.softmax(logits, dim=1)
        target_one_hot = F.one_hot(target, num_classes=self.num_classes).permute(0, 3, 1, 2).float()
        dims = (0, 2, 3)
        inter = torch.sum(probs * target_one_hot, dims)
        denom = torch.sum(probs + target_one_hot, dims)
        dice = (2.0 * inter + self.smooth) / (denom + self.smooth)
        return 1.0 - dice.mean()


class CombinedSegLoss(nn.Module):
    def __init__(self, num_classes: int = 3, dice_weight: float = 1.0) -> None:
        super().__init__()
        self.ce = nn.CrossEntropyLoss()
        self.dice = DiceLoss(num_classes=num_classes)
        self.dice_weight = dice_weight

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return self.ce(logits, target) + self.dice_weight * self.dice(logits, target)


def create_seg_loss(name: str, num_classes: int = 3) -> nn.Module:
    name = name.lower()
    if name == "ce":
        return nn.CrossEntropyLoss()
    if name == "dice":
        return DiceLoss(num_classes=num_classes)
    if name in {"ce_dice", "combined"}:
        return CombinedSegLoss(num_classes=num_classes)
    raise ValueError(f"Unsupported segmentation loss: {name}")

