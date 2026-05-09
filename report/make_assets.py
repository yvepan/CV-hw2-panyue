from __future__ import annotations

import csv
import json
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import shutil

ROOT = Path('.')
ASSETS = ROOT / 'report' / 'assets'
ASSETS.mkdir(parents=True, exist_ok=True)
TASK2_LINE = (0.05, 0.64, 0.95, 0.64)


def read_csv(path: Path) -> list[dict[str, str]]:
    with open(path, newline='', encoding='utf-8') as f:
        return list(csv.DictReader(f))


def plot_task1() -> None:
    runs = {
        'ResNet18 ImageNet': ROOT / 'results/task1/resnet18_imagenet/metrics.csv',
        'ResNet18 random': ROOT / 'results/task1/resnet18_random_init/metrics.csv',
        'ResNet18 CBAM': ROOT / 'results/task1/resnet18_imagenet_cbam/metrics.csv',
        'Swin-T': ROOT / 'results/task1/swin_t_imagenet/metrics.csv',
    }
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    for name, path in runs.items():
        rows = read_csv(path)
        epochs = [int(r['epoch']) for r in rows]
        axes[0].plot(epochs, [float(r['train_loss']) for r in rows], linestyle='--', label=f'{name} train')
        axes[0].plot(epochs, [float(r['val_loss']) for r in rows], label=f'{name} val')
        axes[1].plot(epochs, [float(r['val_acc']) for r in rows], label=name)
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Validation Accuracy')
    for ax in axes:
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(ASSETS / 'task1_val_accuracy.png', dpi=180)
    plt.close(fig)


def plot_task3() -> None:
    runs = {
        'CE': ROOT / 'results/task3/unet_ce/metrics.csv',
        'Dice': ROOT / 'results/task3/unet_dice/metrics.csv',
        'CE+Dice': ROOT / 'results/task3/unet_ce_dice/metrics.csv',
    }
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for name, path in runs.items():
        rows = read_csv(path)
        epochs = [int(r['epoch']) for r in rows]
        axes[0].plot(epochs, [float(r['train_loss']) for r in rows], linestyle='--', label=f'{name} train')
        axes[0].plot(epochs, [float(r['val_loss']) for r in rows], label=f'{name} val')
        axes[1].plot(epochs, [float(r['val_miou']) for r in rows], label=name)
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Validation mIoU')
    for ax in axes:
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(ASSETS / 'task3_curves.png', dpi=180)
    plt.close(fig)


def extract_tracking_frames() -> None:
    cap = cv2.VideoCapture(str(ROOT / 'results/task2/tracking_final/tracked.mp4'))
    id_info = json.loads((ROOT / 'results/task2/tracking_final/id_analysis.json').read_text(encoding='utf-8'))
    frames = [int(f) for f in id_info['frames']]
    for fno in frames:
        cap.set(cv2.CAP_PROP_POS_FRAMES, fno - 1)
        ok, frame = cap.read()
        if ok:
            h, w = frame.shape[:2]
            scale = min(900 / h, 1.0)
            if scale < 1.0:
                frame = cv2.resize(frame, (int(w * scale), int(h * scale)))
            fh, fw = frame.shape[:2]
            cv2.line(
                frame,
                (int(TASK2_LINE[0] * fw), int(TASK2_LINE[1] * fh)),
                (int(TASK2_LINE[2] * fw), int(TASK2_LINE[3] * fh)),
                (0, 0, 255),
                4,
            )
            label = f"frame {fno} | counting line y={TASK2_LINE[1]:.2f}"
            cv2.rectangle(frame, (12, fh - 52), (500, fh - 14), (0, 0, 0), -1)
            cv2.putText(frame, label, (20, fh - 25), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 2, cv2.LINE_AA)
            cv2.imwrite(str(ASSETS / f'task2_frame_{fno}.jpg'), frame)
    cap.release()


def summarize() -> None:
    def load_json(path: str):
        return json.loads((ROOT / path).read_text(encoding='utf-8'))
    summaries = {
        'task1': {
            'resnet18_imagenet': load_json('results/task1/resnet18_imagenet/summary.json'),
            'resnet18_random_init': load_json('results/task1/resnet18_random_init/summary.json'),
            'resnet18_cbam': load_json('results/task1/resnet18_imagenet_cbam/summary.json'),
            'swin_t': load_json('results/task1/swin_t_imagenet/summary.json'),
        },
        'task3': {
            'ce': load_json('results/task3/unet_ce/summary.json'),
            'dice': load_json('results/task3/unet_dice/summary.json'),
            'ce_dice': load_json('results/task3/unet_ce_dice/summary.json'),
        },
        'task2_tracking': load_json('results/task2/tracking_final/summary.json'),
        'task2_id': load_json('results/task2/tracking_final/id_analysis.json'),
    }
    yolo_rows = read_csv(ROOT / 'runs/detect/results/task2/visdrone_yolov8/results.csv')
    m50_idx = 'metrics/mAP50(B)'
    m5095_idx = 'metrics/mAP50-95(B)'
    # Ultralytics includes leading spaces in column names.
    best = max(yolo_rows, key=lambda r: float(r.get(m5095_idx, r.get('       metrics/mAP50-95(B)', 0))))
    summaries['task2_yolo'] = {
        'epochs': int(float(best.get('epoch', best.get('                  epoch', 0)))),
        'best_map50': float(best.get(m50_idx, best.get('       metrics/mAP50(B)', 0))),
        'best_map50_95': float(best.get(m5095_idx, best.get('       metrics/mAP50-95(B)', 0))),
    }
    (ASSETS / 'summary.json').write_text(json.dumps(summaries, ensure_ascii=False, indent=2), encoding='utf-8')


def copy_yolo_assets() -> None:
    yolo_dir = ROOT / 'runs/detect/results/task2/visdrone_yolov8'
    for name, dst in [
        ('BoxPR_curve.png', 'task2_yolo_pr_curve.png'),
    ]:
        src = yolo_dir / name
        if src.exists():
            shutil.copy2(src, ASSETS / dst)


extract_tracking_frames()
summarize()
copy_yolo_assets()
print('tracking frames and supplementary assets generated')
