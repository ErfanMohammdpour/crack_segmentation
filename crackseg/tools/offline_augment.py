from __future__ import annotations

"""Offline 5x augmentation generator (OpenCV-only) for auditing/visualization.

Generates 5 deterministic augmented variants per base image for a chosen split
without affecting official training/evaluation. Uses the same augmenter as
training (rotation±30, scale[0.8,1.2], brightness/contrast±20%, noise 10/255),
applies the same geometry to masks, and keeps masks binary.

CLI:
  python crackseg/tools/offline_augment.py \
      --config crackseg/config.yaml --split train --save ./outputs/offline_aug5_train --seed 42
"""

import argparse
from pathlib import Path
import os
import random
from typing import Dict

import yaml
import numpy as np
import cv2
import torch

from crackseg.data.dataset import CocoCrackDataset
from crackseg.data.utils_io import resolve_paths_from_config, ensure_dir


def load_config(path: Path) -> Dict:
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def set_seed_all(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True, type=str)
    ap.add_argument("--split", required=True, type=str, choices=["train", "valid", "test"])
    ap.add_argument("--save", required=True, type=str)
    ap.add_argument("--seed", default=42, type=int)
    args = ap.parse_args()

    cfg = load_config(Path(args.config))
    data_root, _, _ = resolve_paths_from_config(cfg)

    img_size = int(cfg["IMG_SIZE"])  # Match training resize
    mean = tuple(cfg["MEAN"])  # Not applied here (we save viewable images)
    std = tuple(cfg["STD"])    # Not applied here

    # Build dataset with augment=True so we reuse the same augmenter
    ds = CocoCrackDataset(
        root=Path(data_root),
        split=args.split,
        img_size=img_size,
        mean=mean,
        std=std,
        augment=True,
        aug_multiplier=1,
    )

    out_dir = Path(args.save)
    img_out = out_dir / "images"
    msk_out = out_dir / "masks"
    ensure_dir(img_out)
    ensure_dir(msk_out)

    base_len = ds.base_len  # number of base images

    for j in range(base_len):
        # Get a stable stem for filenames
        stem = f"img_{j:06d}"
        try:
            if ds.coco is not None:
                img_id = ds.img_ids[j]
                info = ds.coco.loadImgs([img_id])[0]
                stem = Path(info.get("file_name", stem)).stem
            else:
                info = ds.images[ds.img_ids[j]]
                stem = Path(info.get("file_name", stem)).stem
        except Exception:
            pass

        # Load original u8 RGB and binary mask
        img_u8, msk_u8 = ds._load_image_and_mask(j)

        for k in range(5):
            # Deterministic per (seed, j, k)
            seed_k = int(args.seed) * 1_000_003 + j * 97 + k
            set_seed_all(seed_k)

            # Convert to 0..1 floats and apply augmentation
            img01 = img_u8.astype(np.float32) / 255.0
            msk01 = (msk_u8 > 0).astype(np.float32)
            assert ds.aug is not None
            img01, msk01 = ds.aug(img01, msk01)

            # Resize to IMG_SIZE with correct interpolation; re-binarize mask
            size = (img_size, img_size)
            img01_r = cv2.resize(img01, size, interpolation=cv2.INTER_LINEAR)
            msk01_r = cv2.resize(msk01, size, interpolation=cv2.INTER_NEAREST)
            msk01_r = (msk01_r > 0.5).astype(np.uint8)

            # Save viewable images (de-normalized to [0,255]) and binary masks
            img_u8_out = np.clip(img01_r * 255.0, 0, 255).astype(np.uint8)
            out_i = img_out / f"{stem}_aug{k}.jpg"
            out_m = msk_out / f"{stem}_aug{k}_mask.png"
            cv2.imwrite(str(out_i), cv2.cvtColor(img_u8_out, cv2.COLOR_RGB2BGR))
            cv2.imwrite(str(out_m), msk01_r * 255)

    print(f"Wrote offline augmented samples to {out_dir}")


if __name__ == "__main__":
    main()

