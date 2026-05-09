from __future__ import annotations

import os
import shutil
from pathlib import Path


VISDRONE_CLASSES = [
    "pedestrian",
    "people",
    "bicycle",
    "car",
    "van",
    "truck",
    "tricycle",
    "awning-tricycle",
    "bus",
    "motor",
]


def _link_or_copy(src: Path, dst: Path) -> None:
    if dst.exists():
        return
    try:
        os.link(src, dst)
    except OSError:
        shutil.copy2(src, dst)


def convert_annotation(annotation_path: Path, image_width: int, image_height: int) -> list[str]:
    labels: list[str] = []
    with open(annotation_path, "r", encoding="utf-8") as f:
        for line in f:
            parts = [p.strip() for p in line.strip().split(",")]
            if len(parts) < 6:
                continue
            x, y, w, h = map(float, parts[:4])
            category = int(parts[5])
            if category <= 0 or category > len(VISDRONE_CLASSES) or w <= 0 or h <= 0:
                continue
            cx = (x + w / 2.0) / image_width
            cy = (y + h / 2.0) / image_height
            nw = w / image_width
            nh = h / image_height
            labels.append(f"{category - 1} {cx:.6f} {cy:.6f} {nw:.6f} {nh:.6f}")
    return labels


def prepare_visdrone(root: str | Path, out: str | Path) -> Path:
    from PIL import Image

    root = Path(root)
    out = Path(out)
    split_map = {
        "VisDrone2019-DET-train": "train",
        "VisDrone2019-DET-val": "val",
        "VisDrone2019-DET-test-dev": "test",
    }
    for src_split, dst_split in split_map.items():
        split_dir = root / src_split
        if not (split_dir / "images").exists() and (split_dir / src_split).exists():
            split_dir = split_dir / src_split
        image_dir = split_dir / "images"
        ann_dir = split_dir / "annotations"
        if not image_dir.exists():
            continue
        dst_img = out / "images" / dst_split
        dst_lbl = out / "labels" / dst_split
        dst_img.mkdir(parents=True, exist_ok=True)
        dst_lbl.mkdir(parents=True, exist_ok=True)
        for image_path in image_dir.glob("*.*"):
            if image_path.suffix.lower() not in {".jpg", ".jpeg", ".png"}:
                continue
            _link_or_copy(image_path, dst_img / image_path.name)
            label_path = dst_lbl / f"{image_path.stem}.txt"
            ann_path = ann_dir / f"{image_path.stem}.txt"
            if ann_path.exists():
                with Image.open(image_path) as img:
                    labels = convert_annotation(ann_path, img.width, img.height)
                label_path.write_text("\n".join(labels) + ("\n" if labels else ""), encoding="utf-8")
            else:
                label_path.write_text("", encoding="utf-8")

    yaml_path = out / "visdrone.yaml"
    names = "\n".join([f"  {i}: {name}" for i, name in enumerate(VISDRONE_CLASSES)])
    yaml_text = (
        f"path: {out.resolve().as_posix()}\n"
        "train: images/train\n"
        "val: images/val\n"
        "test: images/test\n"
        f"names:\n{names}\n"
    )
    yaml_path.write_text(yaml_text, encoding="utf-8")
    return yaml_path
