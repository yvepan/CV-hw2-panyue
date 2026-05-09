from __future__ import annotations

import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from hw2_cv.losses import DiceLoss
from hw2_cv.models import UNet, create_classifier
from hw2_cv.tracking import LineCounter, analyze_id_stability


def main() -> None:
    clf = create_classifier("resnet18", num_classes=37, pretrained=False, attention="se")
    y = clf(torch.randn(2, 3, 224, 224))
    assert y.shape == (2, 37)

    unet = UNet(num_classes=3, base=8)
    logits = unet(torch.randn(2, 3, 64, 64))
    assert logits.shape == (2, 3, 64, 64)
    loss = DiceLoss(num_classes=3)(logits, torch.randint(0, 3, (2, 64, 64)))
    assert torch.isfinite(loss)

    counter = LineCounter((0.5, 0.0), (0.5, 1.0))
    assert counter.update(track_id=1, center=(0.4, 0.5), frame=1) is None
    crossing = counter.update(track_id=1, center=(0.6, 0.5), frame=2)
    assert crossing is not None and crossing["track_id"] == 1

    rows = [
        {"frame": 10, "track_id": 1, "cx": 0.1, "cy": 0.1},
        {"frame": 11, "track_id": 1, "cx": 0.12, "cy": 0.1},
        {"frame": 12, "track_id": 2, "cx": 0.13, "cy": 0.1},
    ]
    report = analyze_id_stability(rows, start_frame=10, num_frames=3)
    assert report["num_frames"] == 3
    print("smoke test passed")


if __name__ == "__main__":
    main()

