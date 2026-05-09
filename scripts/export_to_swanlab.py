from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]


TASK1_RUNS = {
    "resnet18_imagenet": ROOT / "results/task1/resnet18_imagenet/metrics.csv",
    "resnet18_random_init": ROOT / "results/task1/resnet18_random_init/metrics.csv",
    "resnet18_cbam": ROOT / "results/task1/resnet18_imagenet_cbam/metrics.csv",
    "swin_t_imagenet": ROOT / "results/task1/swin_t_imagenet/metrics.csv",
}

TASK3_RUNS = {
    "unet_ce": ROOT / "results/task3/unet_ce/metrics.csv",
    "unet_dice": ROOT / "results/task3/unet_dice/metrics.csv",
    "unet_ce_dice": ROOT / "results/task3/unet_ce_dice/metrics.csv",
}


def read_csv(path: Path) -> list[dict[str, str]]:
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def to_float(value: str) -> float:
    return float(value.strip())


def init_run(swanlab: Any, args: argparse.Namespace, name: str, group: str, config: dict[str, Any]):
    return swanlab.init(
        project=args.project,
        experiment_name=name,
        description="Replayed from local HW2 CSV/result files without retraining.",
        group=group,
        tags=["hw2", "csv-replay", group],
        config=config,
        logdir=args.logdir,
        mode=args.mode,
        reinit=True,
    )


def log_with_step(run: Any, metrics: dict[str, Any], step: int) -> None:
    try:
        run.log(metrics, step=step)
    except TypeError:
        metrics = {"epoch": step, **metrics}
        run.log(metrics)


def finish_run(run: Any, swanlab: Any) -> None:
    if hasattr(run, "finish"):
        run.finish()
    elif hasattr(swanlab, "finish"):
        swanlab.finish()


def export_task1(swanlab: Any, args: argparse.Namespace) -> list[str]:
    exported = []
    for name, path in TASK1_RUNS.items():
        rows = read_csv(path)
        if args.dry_run:
            exported.append(f"task1/{name}: {len(rows)} epochs")
            continue
        run = init_run(
            swanlab,
            args,
            name=f"task1_{name}",
            group="task1_classification",
            config={"task": "classification", "dataset": "Oxford-IIIT Pet", "source_csv": str(path.relative_to(ROOT))},
        )
        for row in rows:
            step = int(row["epoch"])
            log_with_step(
                run,
                {
                    "train/loss": to_float(row["train_loss"]),
                    "train/accuracy": to_float(row["train_acc"]),
                    "val/loss": to_float(row["val_loss"]),
                    "val/accuracy": to_float(row["val_acc"]),
                },
                step,
            )
        finish_run(run, swanlab)
        exported.append(f"task1/{name}: {len(rows)} epochs")
    return exported


def export_task2(swanlab: Any, args: argparse.Namespace) -> list[str]:
    exported = []
    yolo_csv = ROOT / "runs/detect/results/task2/visdrone_yolov8/results.csv"
    yolo_rows = read_csv(yolo_csv)
    if args.dry_run:
        exported.append(f"task2/yolov8: {len(yolo_rows)} epochs")
    else:
        run = init_run(
            swanlab,
            args,
            name="task2_yolov8_visdrone",
            group="task2_detection",
            config={"task": "detection", "dataset": "VisDrone2019-DET", "source_csv": str(yolo_csv.relative_to(ROOT))},
        )
        for row in yolo_rows:
            row = {k.strip(): v for k, v in row.items()}
            step = int(float(row["epoch"]))
            log_with_step(
                run,
                {
                    "train/box_loss": to_float(row["train/box_loss"]),
                    "train/cls_loss": to_float(row["train/cls_loss"]),
                    "train/dfl_loss": to_float(row["train/dfl_loss"]),
                    "val/box_loss": to_float(row["val/box_loss"]),
                    "val/cls_loss": to_float(row["val/cls_loss"]),
                    "val/dfl_loss": to_float(row["val/dfl_loss"]),
                    "metrics/precision": to_float(row["metrics/precision(B)"]),
                    "metrics/recall": to_float(row["metrics/recall(B)"]),
                    "metrics/mAP50": to_float(row["metrics/mAP50(B)"]),
                    "metrics/mAP50-95": to_float(row["metrics/mAP50-95(B)"]),
                },
                step,
            )
        finish_run(run, swanlab)
        exported.append(f"task2/yolov8: {len(yolo_rows)} epochs")

    summary_path = ROOT / "results/task2/tracking_final/summary.json"
    id_path = ROOT / "results/task2/tracking_final/id_analysis.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    id_info = json.loads(id_path.read_text(encoding="utf-8"))
    if args.dry_run:
        exported.append("task2/tracking_final: 1 summary step")
        return exported

    run = init_run(
        swanlab,
        args,
        name="task2_tracking_final",
        group="task2_tracking",
        config={
            "task": "multi-object tracking",
            "video": summary["video"],
            "source_summary": str(summary_path.relative_to(ROOT)),
            "source_id_analysis": str(id_path.relative_to(ROOT)),
        },
    )
    log_with_step(
        run,
        {
            "video/frames": int(summary["frames"]),
            "tracking/records": int(summary["tracks"]),
            "tracking/crossing_count": int(summary["crossing_count"]),
            "id/kept_links": int(id_info["kept_id_links"]),
            "id/lost_links": int(id_info["lost_id_links"]),
            "id/possible_jumps": int(id_info["possible_id_jumps"]),
        },
        0,
    )
    finish_run(run, swanlab)
    exported.append("task2/tracking_final: 1 summary step")
    return exported


def export_task3(swanlab: Any, args: argparse.Namespace) -> list[str]:
    exported = []
    for name, path in TASK3_RUNS.items():
        rows = read_csv(path)
        if args.dry_run:
            exported.append(f"task3/{name}: {len(rows)} epochs")
            continue
        run = init_run(
            swanlab,
            args,
            name=f"task3_{name}",
            group="task3_segmentation",
            config={"task": "segmentation", "dataset": "Oxford-IIIT Pet trimaps", "source_csv": str(path.relative_to(ROOT))},
        )
        for row in rows:
            step = int(row["epoch"])
            log_with_step(
                run,
                {
                    "train/loss": to_float(row["train_loss"]),
                    "train/mIoU": to_float(row["train_miou"]),
                    "val/loss": to_float(row["val_loss"]),
                    "val/mIoU": to_float(row["val_miou"]),
                },
                step,
            )
        finish_run(run, swanlab)
        exported.append(f"task3/{name}: {len(rows)} epochs")
    return exported


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Replay local HW2 metrics into SwanLab without retraining.")
    parser.add_argument("--project", default="cv-hw2")
    parser.add_argument("--mode", default="local", choices=["local", "offline", "cloud", "disabled"])
    parser.add_argument("--logdir", default="swanlog")
    parser.add_argument("--tasks", nargs="+", default=["task1", "task2", "task3"], choices=["task1", "task2", "task3"])
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    swanlab = None
    if not args.dry_run:
        try:
            import swanlab as swanlab_module
        except ImportError as exc:
            raise SystemExit("SwanLab is not installed. Run: python -m pip install swanlab") from exc
        swanlab = swanlab_module

    exported: list[str] = []
    if "task1" in args.tasks:
        exported.extend(export_task1(swanlab, args))
    if "task2" in args.tasks:
        exported.extend(export_task2(swanlab, args))
    if "task3" in args.tasks:
        exported.extend(export_task3(swanlab, args))

    print("SwanLab export plan completed:")
    for item in exported:
        print(f"- {item}")


if __name__ == "__main__":
    main()
