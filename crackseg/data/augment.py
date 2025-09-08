from __future__ import annotations

"""OpenCV-only augmentation utilities for crack segmentation.

Augmentations (train only):
- Random rotation in [-30, +30] degrees
- Random scaling in [0.8, 1.2]
- Brightness/contrast jitter up to 20% (image only)
- Gaussian noise with sigma=10 (image only)

Geometric transforms are applied identically to image and mask.
"""

from dataclasses import dataclass
from typing import Tuple
import random

import cv2
import numpy as np


@dataclass
class TrainAugDocExact:
    """Doc-accurate augmentation at original size using OpenCV only.

    - Geometric (image & mask): rotate [-30, 30], scale [0.8, 1.2] with border reflect.
    - Photometric (image only, uint8): brightness/contrast (±20%), Gaussian noise σ=10.

    Input/Output: uint8 image (H,W,3) and uint8 mask (H,W) in {0,1}.
    """

    def __call__(self, image_u8: np.ndarray, mask_u8: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        h, w = image_u8.shape[:2]

        # Geometric: rotation + scaling at original size
        angle = random.uniform(-30.0, 30.0)
        scale = random.uniform(0.8, 1.2)
        M = cv2.getRotationMatrix2D((w / 2.0, h / 2.0), angle, scale)
        image_u8 = cv2.warpAffine(
            image_u8, M, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT_101
        )
        mask_u8 = cv2.warpAffine(
            mask_u8, M, (w, h), flags=cv2.INTER_NEAREST, borderMode=cv2.BORDER_REFLECT_101
        )
        mask_u8 = (mask_u8 > 0).astype(np.uint8)

        # Photometric: contrast/brightness in uint8, then Gaussian noise via int16 buffer
        alpha = 1.0 + random.uniform(-0.2, 0.2)
        beta = 255.0 * random.uniform(-0.2, 0.2)
        image_u8 = cv2.convertScaleAbs(image_u8, alpha=alpha, beta=beta)
        noise = np.random.normal(0.0, 10.0, image_u8.shape).astype(np.int16)
        image_u8 = np.clip(image_u8.astype(np.int16) + noise, 0, 255).astype(np.uint8)

        return image_u8, mask_u8


# Backward-compatible alias if needed by external code
Augmentor = TrainAugDocExact
