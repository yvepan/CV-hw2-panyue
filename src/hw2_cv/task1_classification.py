from __future__ import annotations

from pathlib import Path

import torch
from torch import nn
from tqdm import tqdm

from .common import AverageMeter, CsvLogger, accuracy, ensure_dir, get_device, save_json, set_seed
from .models import create_classifier
from .pet_data import classification_loaders


def _is_head(name: str) -> bool:
    return name.startswith("fc.") or name.startswith("head.")


def _set_backbone_trainable(model: nn.Module, trainable: bool) -> None:
    for name, param in model.named_parameters():
        if not _is_head(name):
            param.requires_grad = trainable


def _make_optimizer(model: nn.Module, base_lr: float, head_lr: float, weight_decay: float) -> torch.optim.Optimizer:
    base_params = []
    head_params = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if _is_head(name):
            head_params.append(param)
        else:
            base_params.append(param)
    groups = []
    if base_params:
        groups.append({"params": base_params, "lr": base_lr})
    if head_params:
        groups.append({"params": head_params, "lr": head_lr})
    return torch.optim.AdamW(groups, weight_decay=weight_decay)


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
    acc_meter = AverageMeter()
    scaler = torch.cuda.amp.GradScaler(enabled=amp and device.type == "cuda")
    for images, targets in tqdm(loader, leave=False):
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        with torch.set_grad_enabled(training):
            with torch.cuda.amp.autocast(enabled=amp and device.type == "cuda"):
                logits = model(images)
                loss = criterion(logits, targets)
            if training:
                optimizer.zero_grad(set_to_none=True)
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
        bs = images.size(0)
        loss_meter.update(loss.item(), bs)
        acc_meter.update(accuracy(logits.detach(), targets), bs)
    return loss_meter.avg, acc_meter.avg


def train_from_config(cfg: dict) -> dict:
    set_seed(int(cfg.get("seed", 42)))
    out_dir = ensure_dir(cfg["output_dir"])
    train_cfg = cfg["train"]
    model_cfg = cfg["model"]
    device = get_device()

    train_loader, val_loader = classification_loaders(
        cfg["data_root"], train_cfg["image_size"], train_cfg["batch_size"], train_cfg["num_workers"]
    )
    model = create_classifier(
        model_cfg["name"],
        num_classes=37,
        pretrained=bool(model_cfg["pretrained"]),
        attention=model_cfg.get("attention", "none"),
    ).to(device)
    criterion = nn.CrossEntropyLoss(label_smoothing=float(train_cfg.get("label_smoothing", 0.0)))
    amp = bool(train_cfg.get("amp", True))
    freeze_epochs = int(train_cfg.get("freeze_backbone_epochs", 0))

    logger = CsvLogger(out_dir / "metrics.csv", ["epoch", "train_loss", "train_acc", "val_loss", "val_acc"])
    best_acc = 0.0
    best_epoch = 0
    optimizer = None
    scheduler = None
    for epoch in range(1, int(train_cfg["epochs"]) + 1):
        freeze_now = epoch <= freeze_epochs
        _set_backbone_trainable(model, not freeze_now)
        optimizer = _make_optimizer(
            model,
            float(train_cfg["base_lr"]),
            float(train_cfg["head_lr"]),
            float(train_cfg["weight_decay"]),
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, int(train_cfg["epochs"]) - epoch + 1))
        train_loss, train_acc = run_epoch(model, train_loader, criterion, optimizer, device, amp)
        val_loss, val_acc = run_epoch(model, val_loader, criterion, None, device, amp)
        scheduler.step()
        logger.log(
            {
                "epoch": epoch,
                "train_loss": f"{train_loss:.6f}",
                "train_acc": f"{train_acc:.6f}",
                "val_loss": f"{val_loss:.6f}",
                "val_acc": f"{val_acc:.6f}",
            }
        )
        if val_acc > best_acc:
            best_acc = val_acc
            best_epoch = epoch
            torch.save({"model": model.state_dict(), "config": cfg, "val_acc": best_acc}, out_dir / "best.pt")
        print(f"epoch={epoch} train_acc={train_acc:.4f} val_acc={val_acc:.4f} best={best_acc:.4f}")

    summary = {
        "run_name": cfg.get("run_name", Path(out_dir).name),
        "best_epoch": best_epoch,
        "best_val_acc": best_acc,
        "model": model_cfg,
        "train": train_cfg,
    }
    save_json(summary, out_dir / "summary.json")
    return summary

