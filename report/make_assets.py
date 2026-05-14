from __future__ import annotations

import csv
import json
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import shutil
import numpy as np
from PIL import Image, ImageDraw, ImageFont

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


def make_counting_line_example() -> None:
    src = ASSETS / 'task2_frame_250.jpg'
    if src.exists():
        shutil.copy2(src, ASSETS / 'task2_counting_line_example.jpg')


def make_id_switch_assets() -> None:
    tracks_path = ROOT / 'results/task2/tracking_final/tracks.csv'
    if not tracks_path.exists():
        return
    rows = read_csv(tracks_path)
    by_key = {(int(r['frame']), int(r['track_id'])): r for r in rows}
    font = ImageFont.load_default()

    def draw_pair(old_id: int, new_id: int) -> Path:
        panels = []
        for frame, track_id, label in [
            (252, old_id, f'old ID: {old_id}'),
            (253, new_id, f'new ID: {new_id}'),
        ]:
            img_path = ASSETS / f'task2_frame_{frame}.jpg'
            if not img_path.exists() or (frame, track_id) not in by_key:
                continue
            img = Image.open(img_path).convert('RGB')
            r = by_key[(frame, track_id)]
            x1, y1, x2, y2 = [float(r[k]) for k in ('x1', 'y1', 'x2', 'y2')]
            cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
            pad = 90
            w, h = img.size
            crop = (max(0, int(cx - pad)), max(0, int(cy - pad)), min(w, int(cx + pad)), min(h, int(cy + pad)))
            panel = img.crop(crop).resize((320, 320))
            sx = 320 / (crop[2] - crop[0])
            sy = 320 / (crop[3] - crop[1])
            box = [(x1 - crop[0]) * sx, (y1 - crop[1]) * sy, (x2 - crop[0]) * sx, (y2 - crop[1]) * sy]
            d = ImageDraw.Draw(panel)
            for offset in range(4):
                d.rectangle([box[0] - offset, box[1] - offset, box[2] + offset, box[3] + offset], outline=(255, 0, 0))
            d.rectangle([8, 8, 235, 42], fill=(0, 0, 0))
            d.text((14, 17), f'frame {frame} | {label}', fill=(255, 255, 255), font=font)
            panels.append(panel)

        out = ASSETS / f'task2_id_switch_{old_id}_{new_id}.png'
        if len(panels) != 2:
            return out
        canvas = Image.new('RGB', (680, 380), (255, 255, 255))
        canvas.paste(panels[0], (20, 44))
        canvas.paste(panels[1], (340, 44))
        d = ImageDraw.Draw(canvas)
        d.text((20, 14), f'Possible ID switch: {old_id} -> {new_id}', fill=(0, 0, 0), font=font)
        d.line((325, 194, 335, 194), fill=(220, 0, 0), width=4)
        d.polygon([(335, 194), (323, 186), (323, 202)], fill=(220, 0, 0))
        d.text((218, 344), 'same nearby object, changed tracking ID', fill=(180, 0, 0), font=font)
        canvas.save(out)
        return out

    outputs = [draw_pair(3479, 2416), draw_pair(3803, 3549)]
    summary = Image.new('RGB', (720, 820), (255, 255, 255))
    d = ImageDraw.Draw(summary)
    d.text((20, 16), 'Task 2 ID-switch local evidence: frame 252 -> 253', fill=(0, 0, 0), font=font)
    y = 48
    for path in outputs:
        if path.exists():
            im = Image.open(path).convert('RGB')
            summary.paste(im, (20, y))
            y += 390
    summary.save(ASSETS / 'task2_id_switch_summary.png')


def make_task3_segmentation_comparison() -> None:
    data_dir = ROOT / 'datasets/oxford_pet/oxford-iiit-pet'
    test_file = data_dir / 'annotations/test.txt'
    if not test_file.exists():
        return
    runs = ['unet_ce', 'unet_dice', 'unet_ce_dice']
    if not all((ROOT / f'results/task3/{run}/visualizations/sample_0_gt_pred.png').exists() for run in runs):
        return

    names = [line.split()[0] for line in test_file.read_text(encoding='utf-8').splitlines()[:4]]
    colors = np.array([[0, 0, 0], [80, 180, 80], [230, 210, 60]], dtype=np.uint8)
    font = ImageFont.load_default()

    def colorize_mask(path: Path) -> Image.Image:
        arr = np.asarray(Image.open(path).convert('L').resize((192, 192), Image.Resampling.NEAREST), dtype=np.int64) - 1
        arr = np.clip(arr, 0, 2)
        return Image.fromarray(colors[arr], 'RGB')

    def pred_half(run: str, idx: int) -> Image.Image:
        im = Image.open(ROOT / f'results/task3/{run}/visualizations/sample_{idx}_gt_pred.png').convert('RGB')
        w, h = im.size
        return im.crop((w // 2, 0, w, h)).resize((192, 192))

    cols = ['Input', 'Ground Truth', 'CE', 'Dice', 'CE+Dice']
    cell, top, left, gap, row_h = 192, 42, 18, 10, 232
    canvas = Image.new('RGB', (left * 2 + 5 * cell + 4 * gap, top + len(names) * row_h + 18), (255, 255, 255))
    d = ImageDraw.Draw(canvas)
    for c, title in enumerate(cols):
        d.text((left + c * (cell + gap) + 6, 14), title, fill=(0, 0, 0), font=font)
    for i, name in enumerate(names):
        y = top + i * row_h
        image = Image.open(data_dir / f'images/{name}.jpg').convert('RGB')
        image.thumbnail((cell, cell))
        bg = Image.new('RGB', (cell, cell), (245, 245, 245))
        bg.paste(image, ((cell - image.width) // 2, (cell - image.height) // 2))
        panels = [
            bg,
            colorize_mask(data_dir / f'annotations/trimaps/{name}.png'),
            pred_half('unet_ce', i),
            pred_half('unet_dice', i),
            pred_half('unet_ce_dice', i),
        ]
        for c, panel in enumerate(panels):
            x = left + c * (cell + gap)
            canvas.paste(panel, (x, y))
            d.rectangle([x, y, x + cell, y + cell], outline=(210, 210, 210))
        d.text((left, y + cell + 8), name, fill=(50, 50, 50), font=font)
    canvas.save(ASSETS / 'task3_segmentation_comparison.png')


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
make_counting_line_example()
make_id_switch_assets()
make_task3_segmentation_comparison()
summarize()
copy_yolo_assets()
print('tracking frames and supplementary assets generated')
