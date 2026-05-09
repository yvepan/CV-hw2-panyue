from __future__ import annotations

import argparse
from pathlib import Path

import cv2


def prepare_clip(src: str | Path, dst: str | Path, duration: float, fps: float, width: int) -> dict[str, float | int | str]:
    src = Path(src)
    dst = Path(dst)
    cap = cv2.VideoCapture(str(src))
    if not cap.isOpened():
        raise FileNotFoundError(f"Cannot open video: {src}")

    src_fps = cap.get(cv2.CAP_PROP_FPS) or fps
    src_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    src_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    height = int(round(src_h * (width / src_w)))
    height += height % 2
    target_frames = int(round(duration * fps))
    max_src_frames = int(round(duration * src_fps))
    dst.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(str(dst), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))

    out_idx = 0
    src_idx = 0
    next_time = 0.0
    while out_idx < target_frames and src_idx < max_src_frames:
        ok, frame = cap.read()
        if not ok:
            break
        src_time = src_idx / src_fps
        if src_time + 1e-9 >= next_time:
            frame = cv2.resize(frame, (width, height), interpolation=cv2.INTER_AREA)
            writer.write(frame)
            out_idx += 1
            next_time = out_idx / fps
        src_idx += 1

    cap.release()
    writer.release()

    check = cv2.VideoCapture(str(dst))
    out_frames = int(check.get(cv2.CAP_PROP_FRAME_COUNT))
    out_fps = check.get(cv2.CAP_PROP_FPS) or fps
    out_w = int(check.get(cv2.CAP_PROP_FRAME_WIDTH))
    out_h = int(check.get(cv2.CAP_PROP_FRAME_HEIGHT))
    check.release()
    return {
        "source": str(src),
        "output": str(dst),
        "source_fps": float(src_fps),
        "fps": float(out_fps),
        "frames": out_frames,
        "width": out_w,
        "height": out_h,
        "duration": out_frames / out_fps if out_fps else 0.0,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--src", required=True)
    parser.add_argument("--dst", required=True)
    parser.add_argument("--duration", type=float, default=30.0)
    parser.add_argument("--fps", type=float, default=25.0)
    parser.add_argument("--width", type=int, default=1280)
    args = parser.parse_args()
    print(prepare_clip(args.src, args.dst, args.duration, args.fps, args.width))


if __name__ == "__main__":
    main()
