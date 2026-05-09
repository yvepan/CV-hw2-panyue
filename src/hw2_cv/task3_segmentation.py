from __future__ import annotations

from pathlib import Path

import torch
from PIL import Image
from torch import nn
from tqdm import tqdm

from .common import AverageMeter, CsvLogger, ensure_dir, get_device, multiclass_iou, save_json, set_seed
from .losses import create_seg_loss
from .models import UNet
from .pet_data import colorize_mask, segmentation_loaders


def run_epoch(
    model: nn.Module,
    loader: torch.utils.data.DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer | None,
    device: torch.device,
    amp: bool,
) -> tuple[float, float]:
    training = optimizer is not None
    model.train(training)
    loss_meter = AverageMeter()
    miou_meter = AverageMeter()
    scaler = torch.cuda.amp.GradScaler(enabled=amp and device.type == "cuda")
    for images, masks in tqdm(loader, leave=False):
        images = images.to(device, non_blocking=True)
        masks = masks.to(device, non_blocking=True)
        with torch.set_grad_enabled(training):
            with torch.cuda.amp.autocast(enabled=amp and device.type == "cuda"):
                logits = model(images)
                loss = criterion(logits, masks)
            if training:
                optimizer.zero_grad(set_to_none=True)
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
        bs = images.size(0)
        miou, _ = multiclass_iou(logits.detach(), masks, num_classes=3)
        loss_meter.update(loss.item(), bs)
        miou_meter.update(miou, bs)
    return loss_meter.avg, miou_meter.avg


def save_visualization(model: nn.Module, loader: torch.utils.data.DataLoader, out_dir: Path, device: torch.device) -> None:
    model.eval()
    images, masks = next(iter(loader))
    images = images.to(device)
    with torch.no_grad():
        preds = model(images).argmax(dim=1).cpu()
    vis_dir = ensure_dir(out_dir / "visualizations")
    count = min(4, images.size(0))
    for i in range(count):
        gt = colorize_mask(masks[i])
        pred = colorize_mask(preds[i])
        canvas = Image.new("RGB", (gt.width * 2, gt.height))
        canvas.paste(gt, (0, 0))
        canvas.paste(pred, (gt.width, 0))
        canvas.save(vis_dir / f"sample_{i}_gt_pred.png")


def train_from_config(cfg: dict) -> dict:
    set_seed(int(cfg.get("seed", 42)))
    out_dir = ensure_dir(cfg["output_dir"])
    train_cfg = cfg["train"]
    device = get_device()
    train_loader, val_loader = segmentation_loaders(
        cfg["data_root"], train_cfg["image_size"], train_cfg["batch_size"], train_cfg["num_workers"]
    )
    model = UNet(in_channels=3, num_classes=3, base=32).to(device)
    criterion = create_seg_loss(cfg["loss"], num_classes=3)
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(train_cfg["lr"]), weight_decay=float(train_cfg["weight_decay"]))
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=int(train_cfg["epochs"]))
    amp = bool(train_cfg.get("amp", True))
    logger = CsvLogger(out_dir / "metrics.csv", ["epoch", "train_loss", "train_miou", "val_loss", "val_miou"])

    best_miou = 0.0
    best_epoch = 0
    for epoch in range(1, int(train_cfg["epochs"]) + 1):
        train_loss, train_miou = run_epoch(model, train_loader, criterion, optimizer, device, amp)
        val_loss, val_miou = run_epoch(model, val_loader, criterion, None, device, amp)
        scheduler.step()
        logger.log(
            {
                "epoch": epoch,
                "train_loss": f"{train_loss:.6f}",
                "train_miou": f"{train_miou:.6f}",
                "val_loss": f"{val_loss:.6f}",
                "val_miou": f"{val_miou:.6f}",
            }
        )
        if val_miou > best_miou:
            best_miou = val_miou
            best_epoch = epoch
            torch.save({"model": model.state_dict(), "config": cfg, "val_miou": best_miou}, out_dir / "best.pt")
        print(f"epoch={epoch} train_miou={train_miou:.4f} val_miou={val_miou:.4f} best={best_miou:.4f}")

    save_visualization(model, val_loader, out_dir, device)
    summary = {
        "run_name": cfg.get("run_name", Path(out_dir).name),
        "loss": cfg["loss"],
        "best_epoch": best_epoch,
        "best_val_miou": best_miou,
        "train": train_cfg,
    }
    save_json(summary, out_dir / "summary.json")
    return summary

