from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from hw2_cv.visdrone import prepare_visdrone


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True, help="VisDrone dataset root.")
    parser.add_argument("--out", required=True, help="Output YOLO dataset directory.")
    args = parser.parse_args()
    yaml_path = prepare_visdrone(args.root, args.out)
    print(f"wrote {yaml_path}")


if __name__ == "__main__":
    main()

