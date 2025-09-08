from __future__ import annotations

"""Generate a tiny synthetic COCO dataset for CPU tests.

Creates train/valid/test splits with simple line/rectangle cracks.
"""

from pathlib import Path
import json
import random
from typing import Tuple, List

import cv2
import numpy as np


def make_image(w: int, h: int, crack: bool) -> Tuple[np.ndarray, List[List[float]]]:
    img = np.zeros((h, w, 3), dtype=np.uint8)
    img[:] = (30, 30, 30)
    segs: List[List[float]] = []
    if crack:
        # Draw a white thin crack-like polyline
        pts = []
        x = random.randint(w // 4, w // 2)
        y = random.randint(h // 4, h // 2)
        for _ in range(5):
            x += random.randint(-20, 20)
            y += random.randint(10, 30)
            x = max(5, min(w - 5, x))
            y = max(5, min(h - 5, y))
            pts.append([x, y])
        for i in range(len(pts) - 1):
            cv2.line(img, tuple(map(int, pts[i])), tuple(map(int, pts[i + 1])), (230, 230, 230), 2)
        # Polygon around the crack (approx rectangle)
        minx = min(p[0] for p in pts) - 3
        maxx = max(p[0] for p in pts) + 3
        miny = min(p[1] for p in pts) - 3
        maxy = max(p[1] for p in pts) + 3
        poly = [minx, miny, maxx, miny, maxx, maxy, minx, maxy]
        segs.append(poly)
    return img, segs


def build_split(root: Path, split: str, n: int) -> None:
    d = root / split
    d.mkdir(parents=True, exist_ok=True)
    images = []
    annotations = []
    img_id = 1
    ann_id = 1
    for i in range(n):
        crack = random.random() < 0.7
        img, segs = make_image(640, 480, crack)
        fname = f"{split}_{i:03d}.jpg"
        cv2.imwrite(str(d / fname), img)
        images.append({"id": img_id, "file_name": fname, "width": 640, "height": 480})
        for seg in segs:
            # Compute bbox from polygon
            xs = seg[0::2]
            ys = seg[1::2]
            x0, y0 = min(xs), min(ys)
            x1, y1 = max(xs), max(ys)
            bbox = [float(x0), float(y0), float(x1 - x0), float(y1 - y0)]
            annotations.append({
                "id": ann_id,
                "image_id": img_id,
                "category_id": 1,
                "iscrowd": 0,
                "bbox": bbox,
                "segmentation": [list(map(float, seg))],
                "area": float(bbox[2] * bbox[3]),
            })
            ann_id += 1
        img_id += 1
    cats = [{"id": 1, "name": "Crack"}, {"id": 2, "name": "Other"}]
    coco = {"images": images, "annotations": annotations, "categories": cats}
    with (d / "_annotations.coco.json").open("w", encoding="utf-8") as f:
        json.dump(coco, f)


def main() -> None:
    root = Path(__file__).resolve().parents[1] / "sample_data"
    build_split(root, "train", 24)
    build_split(root, "valid", 8)
    build_split(root, "test", 8)
    print(f"Sample dataset written under {root}")


if __name__ == "__main__":
    main()

