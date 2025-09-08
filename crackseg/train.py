from __future__ import annotations

"""Training script for crack segmentation (CPU-only by default)."""

import argparse
import logging
from pathlib import Path
from typing import Dict, Tuple

import yaml
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np

from crackseg.data.dataset import CocoCrackDataset
from crackseg.data.utils_io import ensure_dir, resolve_paths_from_config, set_seed, get_device
from crackseg.losses import make_loss
from crackseg.metrics import compute_metrics
from crackseg.models.unet_mini import UNetMini
from crackseg.models.unet_mini_dropout import UNetMiniDropout
from crackseg.vis import plot_curves, save_triptychs


LOGGER = logging.getLogger("train")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--config", type=str, required=True)
    p.add_argument("--model", type=str, default=None, help="unet_mini | unet_mini_dropout | segformer_lite")
    p.add_argument("--dropout", type=float, default=None, help="Override dropout for dropout model")
    return p.parse_args()


def load_config(path: Path) -> Dict:
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def build_model(name: str, cfg: Dict) -> nn.Module:
    name = name.lower()
    if name == "unet_mini":
        return UNetMini()
    if name == "unet_mini_dropout":
        p = float(cfg.get("DROPOUT", 0.3))
        return UNetMiniDropout(dropout=p)
    if name == "segformer_lite":
        from crackseg.models.segformer_lite import SegFormerLite

        return SegFormerLite(encoder_name=str(cfg.get("ENCODER", "mobilenetv3_large_100")), pretrained=bool(cfg.get("PRETRAINED", 0)))
    raise ValueError(f"Unknown model: {name}")


def make_loader(root: Path, split: str, cfg: Dict, augment: bool) -> DataLoader:
    ds = CocoCrackDataset(
        root=root,
        split=split,
        img_size=int(cfg["IMG_SIZE"]),
        mean=tuple(cfg["MEAN"]),
        std=tuple(cfg["STD"]),
        augment=augment,
        aug_multiplier=int(cfg.get("AUG_MULTIPLIER", 1)),
    )
    loader = DataLoader(
        ds,
        batch_size=int(cfg["BATCH_SIZE"]),
        shuffle=augment,
        num_workers=int(cfg["NUM_WORKERS"]),
        pin_memory=torch.cuda.is_available(),
    )
    return loader


def save_config_snapshot(cfg: Dict, out_dir: Path) -> None:
    ensure_dir(out_dir)
    with (out_dir / "config_snapshot.yaml").open("w", encoding="utf-8") as f:
        yaml.safe_dump(cfg, f, sort_keys=False)


def train_one_epoch(model: nn.Module, loader: DataLoader, loss_fn: nn.Module, optimizer: torch.optim.Optimizer, device: torch.device) -> float:
    model.train()
    running = 0.0
    for images, masks in tqdm(loader, desc="train", leave=False):
        images = images.to(device)
        masks = masks.to(device)
        optimizer.zero_grad(set_to_none=True)
        logits = model(images)
        loss = loss_fn(logits, masks)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=float(cfg_global.get("GRAD_CLIP_NORM", 1.0)))
        optimizer.step()
        running += float(loss.item()) * images.size(0)
    return running / len(loader.dataset)


@torch.no_grad()
def validate(model: nn.Module, loader: DataLoader, loss_fn: nn.Module, device: torch.device, thr: float) -> Tuple[float, Dict[str, float]]:
    model.eval()
    running = 0.0
    metrics_agg = {"IoU": 0.0, "Dice": 0.0, "Precision": 0.0, "Recall": 0.0}
    n = 0
    for images, masks in tqdm(loader, desc="valid", leave=False):
        images = images.to(device)
        masks = masks.to(device)
        logits = model(images)
        loss = loss_fn(logits, masks)
        running += float(loss.item()) * images.size(0)
        m = compute_metrics(logits, masks, thr=thr)
        for k in metrics_agg:
            metrics_agg[k] += m[k] * images.size(0)
        n += images.size(0)
    val_loss = running / max(1, len(loader.dataset))
    for k in metrics_agg:
        metrics_agg[k] = metrics_agg[k] / max(1, n)
    return val_loss, metrics_agg


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    args = parse_args()
    cfg = load_config(Path(args.config))
    global cfg_global
    cfg_global = cfg
    if args.model is not None:
        cfg["MODEL_NAME"] = args.model
    if args.dropout is not None:
        cfg["DROPOUT"] = float(args.dropout)

    data_root, runs_dir, outputs_dir = resolve_paths_from_config(cfg)
    ensure_dir(runs_dir)
    ensure_dir(outputs_dir)

    set_seed(42)
    device = get_device()
    LOGGER.info("Using device: %s", device)

    # Data
    train_loader = make_loader(data_root, "train", cfg, augment=True)
    valid_loader = make_loader(data_root, "valid", cfg, augment=False)

    # Model
    model = build_model(cfg["MODEL_NAME"], cfg).to(device)
    loss_fn = make_loss(cfg["LOSS"]).to(device)

    # Optimizer and scheduler
    if cfg["OPTIMIZER"].lower() == "adamw":
        optimizer = torch.optim.AdamW(model.parameters(), lr=float(cfg["LR"]), weight_decay=float(cfg["WEIGHT_DECAY"]))
    else:
        optimizer = torch.optim.Adam(model.parameters(), lr=float(cfg["LR"]))
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", patience=int(cfg["EARLY_STOPPING_PATIENCE"]) // 2, factor=0.5)

    # Run directory for model
    run_dir = runs_dir / cfg["MODEL_NAME"]
    ensure_dir(run_dir)
    ensure_dir(run_dir / "plots")
    ensure_dir(run_dir / "visuals")
    save_config_snapshot(cfg, run_dir)

    # Training loop
    best_iou = -1.0
    best_path = run_dir / "best.pth"
    early_pat = int(cfg["EARLY_STOPPING_PATIENCE"])
    epochs_no_improve = 0
    E = int(cfg["EPOCHS"])
    thr = float(cfg["THRESHOLD"]) if "THRESHOLD" in cfg else 0.5
    logs_path = run_dir / "logs.csv"
    with logs_path.open("w", encoding="utf-8") as f:
        f.write("epoch,train_loss,val_loss,val_iou,val_dice\n")

    for epoch in range(1, E + 1):
        LOGGER.info("Epoch %d/%d", epoch, E)
        tr_loss = train_one_epoch(model, train_loader, loss_fn, optimizer, device)
        val_loss, val_metrics = validate(model, valid_loader, loss_fn, device, thr=thr)
        scheduler.step(val_loss)

        LOGGER.info("train_loss=%.4f val_loss=%.4f val_iou=%.4f val_dice=%.4f", tr_loss, val_loss, val_metrics["IoU"], val_metrics["Dice"])
        with logs_path.open("a", encoding="utf-8") as f:
            f.write(f"{epoch},{tr_loss:.6f},{val_loss:.6f},{val_metrics['IoU']:.6f},{val_metrics['Dice']:.6f}\n")

        # Best checkpoint by IoU
        if val_metrics["IoU"] > best_iou:
            best_iou = val_metrics["IoU"]
            torch.save({"model": model.state_dict(), "epoch": epoch, "iou": best_iou}, best_path)
            LOGGER.info("Saved new best checkpoint at %s (IoU=%.4f)", best_path, best_iou)
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= early_pat:
                LOGGER.info("Early stopping (no improvement for %d epochs)", early_pat)
                break

    # Finalize: plots and a few qualitative samples
    try:
        plot_curves(logs_path, run_dir / "plots")
    except Exception as e:
        LOGGER.warning("Plotting failed: %s", e)
    try:
        # Reload best weights for visuals
        if best_path.exists():
            state = torch.load(best_path, map_location=device)
            model.load_state_dict(state.get("model", state))
        save_triptychs(model, cfg, run_dir, split="test", n_samples=8)
    except Exception as e:
        LOGGER.warning("Saving triptychs failed: %s", e)

    LOGGER.info("Training complete. Best IoU=%.4f at %s", best_iou, best_path)


if __name__ == "__main__":
    main()
