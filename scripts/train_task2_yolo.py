from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from hw2_cv.task2_yolo import train_yolo_from_yaml


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, help="Path to task2 YOLO YAML config.")
    args = parser.parse_args()
    train_yolo_from_yaml(args.config)


if __name__ == "__main__":
    main()

