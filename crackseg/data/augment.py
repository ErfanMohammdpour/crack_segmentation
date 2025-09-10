from __future__ import annotations

"""OpenCV-only augmentation utilities for crack segmentation (train-only).

Section 4: Data Augmentation (precise, OpenCV-only)
- Rotation: random in [-30, +30] degrees
- Scaling: random in [0.8, 1.2]
- Brightness: random in [-20%, +20%]
- Contrast: random in [-20%, +20%]
- Gaussian noise on image only: sigma = 10/255 (images in [0,1])

Rules
- Build a single affine from rotation + scale around the image center.
- Apply the SAME geometry (one affine) to image and mask.
- Image warp/resize: cv2.INTER_LINEAR; Mask: cv2.INTER_NEAREST.
- Re-binarize mask after any warp/resize to {0,1} exactly.
- No flipping/cropping.

I/O contract for TrainAugDocExact:
- __call__(img_rgb01, mask01) -> (aug_img_rgb01, aug_mask01)
- Inputs are float32 in [0,1] (image HxWx3, mask HxW in {0,1}).
- Outputs are float32; image in [0,1], mask in {0,1}.
"""

from dataclasses import dataclass
from typing import Tuple
import random

import cv2
import numpy as np


@dataclass
class TrainAugDocExact:
    """Doc-accurate augmentation at original size using OpenCV only.

    Parameters
    - img_size: int, used for reference (not resizing here)
    - rot_deg: maximum absolute rotation in degrees (default 30)
    - scale_min, scale_max: scaling range (default [0.8, 1.2])
    - bc_delta: brightness/contrast delta range (±fraction)
    - gauss_sigma_01: Gaussian noise sigma in 0..1 domain (default 10/255)

    __call__(img_rgb01, mask01) expects float32 inputs in [0,1]; returns float32 [0,1] image and {0,1} mask.
    """

    img_size: int = 512
    rot_deg: float = 30.0
    scale_min: float = 0.8
    scale_max: float = 1.2
    bc_delta: float = 0.20
    gauss_sigma_01: float = 10.0 / 255.0

    def __call__(self, img_rgb01: np.ndarray, mask01: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        assert img_rgb01.dtype in (np.float32, np.float64), "image must be float"
        assert mask01.dtype in (np.float32, np.float64, np.uint8), "mask must be float or uint8"
        img = img_rgb01.astype(np.float32)
        msk = mask01.astype(np.float32)
        h, w = img.shape[:2]

        # Geometric: rotation + scaling (single affine), same for image and mask
        angle = random.uniform(-float(self.rot_deg), float(self.rot_deg))
        scale = random.uniform(float(self.scale_min), float(self.scale_max))
        M = cv2.getRotationMatrix2D((w / 2.0, h / 2.0), angle, scale)
        img = cv2.warpAffine(img, M, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT_101)
        msk = cv2.warpAffine(msk, M, (w, h), flags=cv2.INTER_NEAREST, borderMode=cv2.BORDER_REFLECT_101)
        # Re-binarize mask strictly to {0,1}
        msk = (msk > 0.5).astype(np.float32)

        # Photometric: contrast then brightness on image (0..1), then Gaussian noise
        alpha = 1.0 + random.uniform(-self.bc_delta, self.bc_delta)  # contrast
        beta = random.uniform(-self.bc_delta, self.bc_delta)         # brightness (additive)
        img = np.clip(alpha * img + beta, 0.0, 1.0)
        if self.gauss_sigma_01 > 0:
            noise = np.random.normal(0.0, float(self.gauss_sigma_01), img.shape).astype(np.float32)
            img = np.clip(img + noise, 0.0, 1.0)

        return img.astype(np.float32), msk.astype(np.float32)


# Backward-compatible alias if needed by external code
Augmentor = TrainAugDocExact

