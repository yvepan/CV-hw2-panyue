from __future__ import annotations

import csv
from pathlib import Path
from typing import Any

from .common import ensure_dir, load_config, save_json
from .tracking import LineCounter, analyze_id_stability


def train_yolo_from_config(cfg: dict[str, Any]) -> None:
    try:
        from ultralytics import YOLO
    except ImportError as exc:
        raise ImportError("YOLOv8 training requires ultralytics. Install requirements.txt first.") from exc

    model = YOLO(cfg["model"])
    model.train(
        data=cfg["data_yaml"],
        epochs=int(cfg["epochs"]),
        imgsz=int(cfg["imgsz"]),
        batch=int(cfg["batch"]),
        project=cfg["output_dir"],
        name=cfg["run_name"],
        device=cfg.get("device", "auto"),
        workers=int(cfg.get("workers", 4)),
        patience=int(cfg.get("patience", 15)),
        optimizer=cfg.get("optimizer", "AdamW"),
        lr0=float(cfg.get("lr0", 0.001)),
        weight_decay=float(cfg.get("weight_decay", 0.0005)),
    )


def track_video(
    weights: str | Path,
    video: str | Path,
    out: str | Path,
    line: tuple[float, float, float, float],
    analyze_start: int,
    analyze_len: int,
    tracker: str = "bytetrack.yaml",
    conf: float = 0.25,
) -> dict[str, Any]:
    try:
        import cv2
        from ultralytics import YOLO
    except ImportError as exc:
        raise ImportError("Tracking requires ultralytics and opencv-python. Install requirements.txt first.") from exc

    out_dir = ensure_dir(out)
    model = YOLO(str(weights))
    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        raise FileNotFoundError(f"Cannot open video: {video}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    writer = cv2.VideoWriter(
        str(out_dir / "tracked.mp4"),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (width, height),
    )
    line_a = (line[0], line[1])
    line_b = (line[2], line[3])
    counter = LineCounter(line_a, line_b)

    track_rows: list[dict[str, Any]] = []
    crossing_rows: list[dict[str, Any]] = []
    frame_idx = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frame_idx += 1
        result = model.track(frame, persist=True, tracker=tracker, conf=conf, verbose=False)[0]
        boxes = result.boxes
        if boxes is not None and boxes.id is not None:
            xyxy = boxes.xyxy.cpu().numpy()
            ids = boxes.id.cpu().numpy().astype(int)
            cls = boxes.cls.cpu().numpy().astype(int)
            confs = boxes.conf.cpu().numpy()
            for box, tid, cls_id, score in zip(xyxy, ids, cls, confs):
                x1, y1, x2, y2 = map(float, box)
                cx = ((x1 + x2) / 2.0) / width
                cy = ((y1 + y2) / 2.0) / height
                row = {
                    "frame": frame_idx,
                    "track_id": int(tid),
                    "class_id": int(cls_id),
                    "conf": float(score),
                    "x1": x1,
                    "y1": y1,
                    "x2": x2,
                    "y2": y2,
                    "cx": cx,
                    "cy": cy,
                }
                track_rows.append(row)
                crossing = counter.update(int(tid), (cx, cy), frame_idx)
                if crossing:
                    crossing_rows.append(crossing)
                color = (40 + (int(tid) * 37) % 200, 180, 80)
                cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
                cv2.putText(
                    frame,
                    f"id={tid} c={cls_id} {score:.2f}",
                    (int(x1), max(20, int(y1) - 5)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.55,
                    color,
                    2,
                )
        p1 = (int(line_a[0] * width), int(line_a[1] * height))
        p2 = (int(line_b[0] * width), int(line_b[1] * height))
        cv2.line(frame, p1, p2, (0, 0, 255), 2)
        cv2.putText(frame, f"crossing count: {counter.count}", (20, 35), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)
        writer.write(frame)

    cap.release()
    writer.release()

    tracks_csv = out_dir / "tracks.csv"
    with open(tracks_csv, "w", newline="", encoding="utf-8") as f:
        fieldnames = ["frame", "track_id", "class_id", "conf", "x1", "y1", "x2", "y2", "cx", "cy"]
        writer_csv = csv.DictWriter(f, fieldnames=fieldnames)
        writer_csv.writeheader()
        writer_csv.writerows(track_rows)

    crossings_csv = out_dir / "crossings.csv"
    with open(crossings_csv, "w", newline="", encoding="utf-8") as f:
        fieldnames = ["frame", "track_id", "cx", "cy", "direction", "total"]
        writer_csv = csv.DictWriter(f, fieldnames=fieldnames)
        writer_csv.writeheader()
        writer_csv.writerows(crossing_rows)

    stability = analyze_id_stability(track_rows, analyze_start, analyze_len)
    save_json(stability, out_dir / "id_analysis.json")
    summary = {
        "video": str(video),
        "weights": str(weights),
        "frames": frame_idx,
        "tracks": len(track_rows),
        "crossing_count": counter.count,
        "outputs": {
            "video": str(out_dir / "tracked.mp4"),
            "tracks_csv": str(tracks_csv),
            "crossings_csv": str(crossings_csv),
            "id_analysis_json": str(out_dir / "id_analysis.json"),
        },
    }
    save_json(summary, out_dir / "summary.json")
    return summary


def train_yolo_from_yaml(path: str | Path) -> None:
    train_yolo_from_config(load_config(path))

