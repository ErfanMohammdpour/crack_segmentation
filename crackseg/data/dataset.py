from __future__ import annotations

"""COCO-based dataset for crack segmentation with binary masks.

Rules:
- Positive = any category whose name contains 'crack' (case-insensitive).
- Mask = union of all positive annotations via COCO annToMask; if missing segmentation, fall back to bbox.
- Images are normalized with ImageNet mean/std; masks are uint8 {0,1}.
- Resize: image bilinear; mask nearest.
- Returns tensors: image float32 [C,H,W], mask float32 [1,H,W].
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import logging

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

from .augment import TrainAugDocExact
from .utils_io import find_coco_annotation_file, load_json


LOGGER = logging.getLogger(__name__)


def _try_load_coco(json_path: Path):
    """Try to load pycocotools COCO, return COCO or None if not available."""
    try:
        from pycocotools.coco import COCO  # type: ignore
        return COCO(str(json_path))
    except Exception:
        return None


def _positive_category_ids(coco) -> List[int]:
    cats = coco.loadCats(coco.getCatIds())
    pos_ids = [c["id"] for c in cats if "crack" in c.get("name", "").lower()]
    return pos_ids


def _bbox_to_mask(bbox: List[float], h: int, w: int) -> np.ndarray:
    x, y, bw, bh = bbox
    x0 = int(max(0, np.floor(x)))
    y0 = int(max(0, np.floor(y)))
    x1 = int(min(w, np.ceil(x + bw)))
    y1 = int(min(h, np.ceil(y + bh)))
    m = np.zeros((h, w), dtype=np.uint8)
    m[y0:y1, x0:x1] = 1
    return m


@dataclass
class CocoCrackDataset(Dataset):
    root: Path
    split: str
    img_size: int
    mean: Tuple[float, float, float]
    std: Tuple[float, float, float]
    augment: bool = False
    aug_multiplier: int = 1

    def __post_init__(self) -> None:
        self.split_dir = self.root / self.split
        json_path = find_coco_annotation_file(self.split_dir)
        self.coco = _try_load_coco(json_path)
        self.json_data: Optional[Dict] = None
        if self.coco is None:
            LOGGER.warning(
                "pycocotools not available. Falling back to cv2-based polygon rasterization."
            )
            self.json_data = load_json(json_path)
            self.images = {im["id"]: im for im in self.json_data.get("images", [])}
            self.anns_by_img: Dict[int, List[Dict]] = {}
            for ann in self.json_data.get("annotations", []):
                self.anns_by_img.setdefault(ann["image_id"], []).append(ann)
            self.cats = {c["id"]: c for c in self.json_data.get("categories", [])}
            self.pos_cat_ids = [cid for cid, c in self.cats.items() if "crack" in c.get("name", "").lower()]
            self.img_ids = list(self.images.keys())
        else:
            self.img_ids: List[int] = self.coco.getImgIds()
            self.pos_cat_ids = _positive_category_ids(self.coco)
            if not self.pos_cat_ids:
                LOGGER.warning("No positive categories (containing 'crack') found in categories.")
        self.base_len = len(self.img_ids)
        self.total_len = self.base_len * max(1, int(self.aug_multiplier))
        if self.augment and self.aug_multiplier > 1:
            LOGGER.info("Training set augmented with multiplier=%d", self.aug_multiplier)
        # OpenCV-only augmenter operating in 0..1 float domain
        self.aug = TrainAugDocExact(img_size=int(self.img_size)) if self.augment else None

    def __len__(self) -> int:
        return self.total_len

    def _load_image_and_mask(self, idx: int) -> Tuple[np.ndarray, np.ndarray]:
        img_id = self.img_ids[idx]
        if self.coco is not None:
            img_info = self.coco.loadImgs([img_id])[0]
            img_path = self.split_dir / img_info["file_name"]
            image_bgr = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
            if image_bgr is None:
                raise FileNotFoundError(f"Failed to read image: {img_path}")
            image_u8 = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
            h, w = image_u8.shape[:2]
            anns = self.coco.loadAnns(self.coco.getAnnIds(imgIds=[img_id]))
            mask = np.zeros((h, w), dtype=np.uint8)
            for ann in anns:
                if ann.get("category_id") not in self.pos_cat_ids:
                    continue
                try:
                    m = self.coco.annToMask(ann).astype(np.uint8)
                except Exception:
                    bbox = ann.get("bbox", None)
                    if bbox is None:
                        continue
                    m = _bbox_to_mask(bbox, h, w)
                mask = np.maximum(mask, m)
            mask = (mask > 0).astype(np.uint8)
            return image_u8, mask
        else:
            assert self.json_data is not None
            img_info = self.images[img_id]
            img_path = self.split_dir / img_info["file_name"]
            image_bgr = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
            if image_bgr is None:
                raise FileNotFoundError(f"Failed to read image: {img_path}")
            image_u8 = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
            h, w = image_u8.shape[:2]
            mask = np.zeros((h, w), dtype=np.uint8)
            for ann in self.anns_by_img.get(img_id, []):
                if ann.get("category_id") not in self.pos_cat_ids:
                    continue
                seg = ann.get("segmentation", None)
                if isinstance(seg, list) and len(seg) > 0:
                    # Polygon(s)
                    polys = []
                    for poly in seg:
                        pts = np.array(poly, dtype=np.float32).reshape(-1, 2)
                        polys.append(pts.astype(np.int32))
                    m = np.zeros((h, w), dtype=np.uint8)
                    cv2.fillPoly(m, polys, 1)
                else:
                    bbox = ann.get("bbox", None)
                    if bbox is None:
                        continue
                    m = _bbox_to_mask(bbox, h, w)
                mask = np.maximum(mask, m)
            mask = (mask > 0).astype(np.uint8)
            return image_u8, mask

    def _resize(self, image_u8: np.ndarray, mask_u8: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        size = (int(self.img_size), int(self.img_size))
        image_r = cv2.resize(image_u8, size, interpolation=cv2.INTER_LINEAR)
        mask_r = cv2.resize(mask_u8, size, interpolation=cv2.INTER_NEAREST)
        mask_r = (mask_r > 0).astype(np.uint8)
        return image_r, mask_r

    def _normalize(self, image_f32: np.ndarray) -> np.ndarray:
        mean = np.array(self.mean, dtype=np.float32).reshape(1, 1, 3)
        std = np.array(self.std, dtype=np.float32).reshape(1, 1, 3)
        return (image_f32 - mean) / std

    def __getitem__(self, index: int):
        base_index = index % self.base_len
        # 1) Load original image RGB uint8 and binary mask uint8 {0,1}
        image_u8, mask_u8 = self._load_image_and_mask(base_index)

        # 2) Augmentation at original size (OpenCV-only), in 0..1 float domain
        if self.aug is not None:
            img01 = image_u8.astype(np.float32) / 255.0
            msk01 = (mask_u8 > 0).astype(np.float32)
            img01, msk01 = self.aug(img01, msk01)
            # 3) Resize to target with correct interpolation, then re-binarize mask
            size = (int(self.img_size), int(self.img_size))
            img01 = cv2.resize(img01, size, interpolation=cv2.INTER_LINEAR)
            msk01 = cv2.resize(msk01, size, interpolation=cv2.INTER_NEAREST)
            msk01 = (msk01 > 0.5).astype(np.float32)
            x = self._normalize(img01)
            x = np.transpose(x, (2, 0, 1))
            y = msk01[None, ...].astype(np.float32)
            return torch.from_numpy(x), torch.from_numpy(y)

        # 3) No augmentation: resize first
        image_u8, mask_u8 = self._resize(image_u8, mask_u8)

        # 4) Normalize and tensorize in 0..1
        x = image_u8.astype(np.float32) / 255.0
        x = self._normalize(x)
        x = np.transpose(x, (2, 0, 1))
        y = mask_u8.astype(np.float32)[None, ...]
        return torch.from_numpy(x), torch.from_numpy(y)
