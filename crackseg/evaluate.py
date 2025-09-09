from __future__ import annotations

"""Evaluation script for crack segmentation on test split."""

import argparse
import json
import logging
from pathlib import Path
from typing import Dict

import yaml
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from crackseg.data.dataset import CocoCrackDataset
from crackseg.data.utils_io import ensure_dir, resolve_paths_from_config, set_seed, get_device
from crackseg.metrics import compute_metrics
from crackseg.models.unet_mini import UNetMini
from crackseg.models.unet_mini_dropout import UNetMiniDropout
from crackseg.models.scratch_ed import ScratchED
from crackseg.models.scratch_ed_plus import ScratchEDPlus


LOGGER = logging.getLogger("evaluate")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--config", type=str, required=True)
    p.add_argument("--weights", type=str, required=True)
    p.add_argument("--model", type=str, default=None)
    p.add_argument("--dropout", type=float, default=None)
    return p.parse_args()


def load_config(path: Path) -> Dict:
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def build_model(name: str, cfg: Dict) -> torch.nn.Module:
    name = name.lower()
    if name == "unet_mini":
        return UNetMini()
    if name == "unet_mini_dropout":
        p = float(cfg.get("DROPOUT", 0.3))
        return UNetMiniDropout(dropout=p)
    if name == "segformer_lite":
        from crackseg.models.segformer_lite import SegFormerLite

        return SegFormerLite(encoder_name=str(cfg.get("ENCODER", "mobilenetv3_large_100")), pretrained=False)
    if name == "scratch_ed":
        base_ch = int(cfg.get("BASE_CHANNELS", 32))
        return ScratchED(base_ch=base_ch)
    if name == "scratch_ed_plus":
        base_ch = int(cfg.get("BASE_CHANNELS", 32))
        rates = list(cfg.get("ASPP_RATES", [1, 6, 12, 18]))
        return ScratchEDPlus(base_ch=base_ch, aspp_rates=rates)
    raise ValueError(f"Unknown model: {name}")


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    args = parse_args()
    cfg = load_config(Path(args.config))
    if args.model is not None:
        cfg["MODEL_NAME"] = args.model
    if args.dropout is not None:
        cfg["DROPOUT"] = float(args.dropout)

    set_seed(42)
    device = get_device()
    data_root, runs_dir, outputs_dir = resolve_paths_from_config(cfg)
    ensure_dir(Path(outputs_dir))

    ds = CocoCrackDataset(
        root=data_root,
        split="test",
        img_size=int(cfg["IMG_SIZE"]),
        mean=tuple(cfg["MEAN"]),
        std=tuple(cfg["STD"]),
        augment=False,
        aug_multiplier=1,
    )
    loader = DataLoader(ds, batch_size=1, shuffle=False, num_workers=int(cfg["NUM_WORKERS"]))

    model = build_model(cfg["MODEL_NAME"], cfg).to(device)
    ckpt = torch.load(args.weights, map_location=device)
    state = ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt
    model.load_state_dict(state)
    model.eval()

    thr = float(cfg.get("THRESHOLD", 0.5))
    agg = {"IoU": 0.0, "Dice": 0.0, "Precision": 0.0, "Recall": 0.0}
    n = 0
    with torch.no_grad():
        for images, masks in tqdm(loader, desc="test"):
            images = images.to(device)
            masks = masks.to(device)
            logits = model(images)
            m = compute_metrics(logits, masks, thr=thr)
            for k in agg:
                agg[k] += m[k]
            n += 1
    for k in agg:
        agg[k] = agg[k] / max(1, n)

    print("Test metrics:")
    for k, v in agg.items():
        print(f"{k}: {v:.4f}")

    out_path = Path(outputs_dir) / "metrics_test.json"
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(agg, f, indent=2)
    LOGGER.info("Saved test metrics to %s", out_path)


if __name__ == "__main__":
    main()
