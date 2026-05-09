from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from hw2_cv.common import load_config
from hw2_cv.task1_classification import train_from_config


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, help="Path to YAML config.")
    args = parser.parse_args()
    cfg = load_config(args.config)
    train_from_config(cfg)


if __name__ == "__main__":
    main()

