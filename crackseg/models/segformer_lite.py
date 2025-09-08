from __future__ import annotations

"""Optional SegFormer-Lite style model using timm backbones.

This file is only used if `timm` is installed and the user selects it.
To keep CPU-only users lightweight, the default config uses UNet variants.
"""

from typing import List
import torch
import torch.nn as nn


class SegFormerLite(nn.Module):
    def __init__(self, encoder_name: str = "mobilenetv3_large_100", pretrained: bool = False):
        super().__init__()
        try:
            import timm  # type: ignore
        except Exception as e:
            raise ImportError(
                "timm is required for segformer_lite. Install with `pip install timm` or switch to unet_mini."
            ) from e

        self.encoder = timm.create_model(encoder_name, features_only=True, pretrained=pretrained, in_chans=3)
        feat_chs: List[int] = self.encoder.feature_info.channels()
        self.proj = nn.ModuleList([nn.Conv2d(c, 64, kernel_size=1) for c in feat_chs])
        self.fuse = nn.Sequential(
            nn.Conv2d(64 * len(feat_chs), 128, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )
        self.head = nn.Conv2d(64, 1, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feats = self.encoder(x)
        # Upsample all to the highest resolution
        h, w = feats[0].shape[-2:]
        ups = []
        for f, p in zip(feats, self.proj):
            f = p(f)
            if f.shape[-2:] != (h, w):
                f = torch.nn.functional.interpolate(f, size=(h, w), mode="bilinear", align_corners=False)
            ups.append(f)
        x = torch.cat(ups, dim=1)
        x = self.fuse(x)
        x = self.head(x)
        # Final logits at highest resolution, caller may upsample to input size
        return x

