from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from hw2_cv.task2_yolo import track_video


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", required=True)
    parser.add_argument("--video", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--line", nargs=4, type=float, required=True, metavar=("X1", "Y1", "X2", "Y2"))
    parser.add_argument("--analyze-start", type=int, required=True)
    parser.add_argument("--analyze-len", type=int, default=4)
    parser.add_argument("--tracker", default="bytetrack.yaml")
    parser.add_argument("--conf", type=float, default=0.25)
    args = parser.parse_args()
    summary = track_video(
        args.weights,
        args.video,
        args.out,
        tuple(args.line),
        args.analyze_start,
        args.analyze_len,
        tracker=args.tracker,
        conf=args.conf,
    )
    print(summary)


if __name__ == "__main__":
    main()

