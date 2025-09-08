from __future__ import annotations

"""Inference script for crack segmentation.

Saves binary masks and overlays at original resolution.
"""

import argparse
from pathlib import Path
from typing import Dict, List
import logging

import yaml
import torch
import numpy as np
import cv2

from crackseg.data.utils_io import resolve_paths_from_config, ensure_dir, get_device
from crackseg.models.unet_mini import UNetMini
from crackseg.models.unet_mini_dropout import UNetMiniDropout


LOGGER = logging.getLogger("infer")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--config", type=str, required=True)
    p.add_argument("--weights", type=str, required=True)
    p.add_argument("--input", type=str, required=True, help="Image path or folder")
    p.add_argument("--save", type=str, required=True)
    p.add_argument("--model", type=str, default=None)
    p.add_argument("--dropout", type=float, default=None)
    p.add_argument("--threshold", type=float, default=None)
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
    raise ValueError(f"Unknown model: {name}")


def preprocess_image(img: np.ndarray, img_size: int, mean, std) -> np.ndarray:
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    img = cv2.resize(img, (img_size, img_size), interpolation=cv2.INTER_LINEAR)
    mean = np.array(mean, dtype=np.float32).reshape(1, 1, 3)
    std = np.array(std, dtype=np.float32).reshape(1, 1, 3)
    img = (img - mean) / std
    return img


def overlay_mask(image_bgr: np.ndarray, mask: np.ndarray, alpha: float = 0.5) -> np.ndarray:
    overlay = image_bgr.copy()
    red = np.zeros_like(image_bgr)
    red[:, :, 2] = 255
    mask3 = np.repeat(mask[:, :, None], 3, axis=2).astype(bool)
    overlay[mask3] = (alpha * red + (1 - alpha) * overlay)[mask3]
    return overlay


def gather_inputs(path: Path) -> List[Path]:
    if path.is_file():
        return [path]
    if path.is_dir():
        exts = {".jpg", ".jpeg", ".png", ".bmp"}
        return sorted([p for p in path.rglob("*") if p.suffix.lower() in exts])
    raise FileNotFoundError(f"Input not found: {path}")


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    args = parse_args()
    cfg = load_config(Path(args.config))
    if args.model is not None:
        cfg["MODEL_NAME"] = args.model
    if args.dropout is not None:
        cfg["DROPOUT"] = float(args.dropout)
    if args.threshold is not None:
        cfg["THRESHOLD"] = float(args.threshold)

    data_root, runs_dir, outputs_dir = resolve_paths_from_config(cfg)
    device = get_device()
    LOGGER.info("Using device: %s", device)

    model = build_model(cfg["MODEL_NAME"], cfg).to(device)
    ckpt = torch.load(args.weights, map_location=device)
    state = ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt
    model.load_state_dict(state)
    model.eval()

    img_size = int(cfg["IMG_SIZE"])
    mean, std = cfg["MEAN"], cfg["STD"]
    thr = float(cfg.get("THRESHOLD", 0.5))

    inp = Path(args.input)
    save_dir = Path(args.save)
    ensure_dir(save_dir)

    inputs = gather_inputs(inp)
    for p in inputs:
        bgr = cv2.imread(str(p), cv2.IMREAD_COLOR)
        if bgr is None:
            continue
        h0, w0 = bgr.shape[:2]
        img = preprocess_image(bgr, img_size, mean, std)
        x = torch.from_numpy(img.transpose(2, 0, 1)).unsqueeze(0).float().to(device)
        with torch.no_grad():
            logits = model(x)
            probs = torch.sigmoid(logits)[0, 0].cpu().numpy()
        mask = (probs >= thr).astype(np.uint8)
        mask = cv2.resize(mask, (w0, h0), interpolation=cv2.INTER_NEAREST)
        overlay = overlay_mask(bgr, mask * 255)

        out_mask = save_dir / f"{p.stem}_mask.png"
        out_overlay = save_dir / f"{p.stem}_overlay.png"
        cv2.imwrite(str(out_mask), mask * 255)
        cv2.imwrite(str(out_overlay), overlay)

    LOGGER.info("Saved results to %s", save_dir)


if __name__ == "__main__":
    main()
