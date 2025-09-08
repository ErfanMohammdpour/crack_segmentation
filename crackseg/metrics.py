from __future__ import annotations

"""Metrics for binary segmentation: IoU, Dice, Precision, Recall."""

from typing import Dict
import torch


def _threshold_logits(logits: torch.Tensor, thr: float = 0.5) -> torch.Tensor:
    probs = torch.sigmoid(logits)
    return (probs >= thr).float()


def compute_metrics(logits: torch.Tensor, targets: torch.Tensor, thr: float = 0.5, eps: float = 1e-7) -> Dict[str, float]:
    preds = _threshold_logits(logits, thr)
    preds = preds.view(preds.size(0), -1)
    targets = targets.view(targets.size(0), -1)
    tp = (preds * targets).sum(dim=1)
    fp = (preds * (1 - targets)).sum(dim=1)
    fn = ((1 - preds) * targets).sum(dim=1)
    tn = ((1 - preds) * (1 - targets)).sum(dim=1)

    iou = (tp + eps) / (tp + fp + fn + eps)
    dice = (2 * tp + eps) / (2 * tp + fp + fn + eps)
    precision = (tp + eps) / (tp + fp + eps)
    recall = (tp + eps) / (tp + fn + eps)

    return {
        "IoU": float(iou.mean().item()),
        "Dice": float(dice.mean().item()),
        "Precision": float(precision.mean().item()),
        "Recall": float(recall.mean().item()),
    }

