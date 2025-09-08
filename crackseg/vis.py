from __future__ import annotations

"""Visualization utilities: curves and qualitative triptychs."""

from pathlib import Path
from typing import List
import csv
import random

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader

from crackseg.data.dataset import CocoCrackDataset


def plot_curves(log_csv: Path, out_dir: Path) -> None:
    epochs, tr, vl, iou, dice = [], [], [], [], []
    with log_csv.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            epochs.append(int(row["epoch"]))
            tr.append(float(row["train_loss"]))
            vl.append(float(row["val_loss"]))
            iou.append(float(row["val_iou"]))
            dice.append(float(row["val_dice"]))
    out_dir.mkdir(parents=True, exist_ok=True)

    plt.figure()
    plt.plot(epochs, tr, label="train_loss")
    plt.plot(epochs, vl, label="val_loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_dir / "loss_curves.png")
    plt.close()

    plt.figure()
    plt.plot(epochs, iou, label="IoU")
    plt.plot(epochs, dice, label="Dice")
    plt.xlabel("Epoch")
    plt.ylabel("Score")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_dir / "metrics_curves.png")
    plt.close()


def save_triptychs(model: torch.nn.Module, cfg: dict, run_dir: Path, split: str = "test", n_samples: int = 8) -> None:
    ds = CocoCrackDataset(
        root=Path(cfg["DATA_ROOT"]),
        split=split,
        img_size=int(cfg["IMG_SIZE"]),
        mean=tuple(cfg["MEAN"]),
        std=tuple(cfg["STD"]),
        augment=False,
        aug_multiplier=1,
    )
    idxs = random.sample(range(len(ds)), k=min(n_samples, len(ds)))
    out_dir = run_dir / "visuals"
    out_dir.mkdir(parents=True, exist_ok=True)
    model.eval()
    device = next(model.parameters()).device
    with torch.no_grad():
        for i in idxs:
            img_t, mask_t = ds[i]
            logits = model(img_t.unsqueeze(0).to(device))
            prob = torch.sigmoid(logits)[0, 0].cpu().numpy()
            pred = (prob >= float(cfg.get("THRESHOLD", 0.5))).astype(np.uint8)

            # De-normalize for display
            img = img_t.numpy().transpose(1, 2, 0)
            mean = np.array(cfg["MEAN"], dtype=np.float32)
            std = np.array(cfg["STD"], dtype=np.float32)
            disp = np.clip(img * std + mean, 0, 1)

            fig, axes = plt.subplots(1, 3, figsize=(12, 4))
            axes[0].imshow(disp)
            axes[0].set_title("Input")
            axes[1].imshow(mask_t[0].numpy(), cmap="gray")
            axes[1].set_title("Ground Truth")
            axes[2].imshow(disp)
            axes[2].imshow(pred, cmap="Reds", alpha=0.5)
            axes[2].set_title("Prediction")
            for ax in axes:
                ax.axis("off")
            plt.tight_layout()
            plt.savefig(out_dir / f"sample_{i}.png")
            plt.close()
