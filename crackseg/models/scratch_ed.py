from __future__ import annotations

"""From-scratch encoder–decoder segmentation model without U-Net skips.

Design goals:
- Pure PyTorch, lightweight (~1–3M params depending on `base_ch`).
- Encoder: 3 downsampling stages via stride-2 convs (no MaxPool), each with two Conv-BN-ReLU.
- Bottleneck: two Conv-BN-ReLU blocks.
- Decoder: bilinear upsample + two Conv-BN-ReLU per stage; no cross-scale skip connections.
- Head: 1×1 conv to a single logit channel.
"""

from typing import Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F


def conv3x3(in_ch: int, out_ch: int, stride: int = 1) -> nn.Conv2d:
    return nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=stride, padding=1, bias=False)


class ConvBNReLU(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, stride: int = 1):
        super().__init__()
        self.conv = conv3x3(in_ch, out_ch, stride)
        self.bn = nn.BatchNorm2d(out_ch)
        self.act = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(x)
        x = self.bn(x)
        return self.act(x)


class DoubleConv(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, stride: int = 1):
        super().__init__()
        self.block = nn.Sequential(
            ConvBNReLU(in_ch, out_ch, stride=stride),
            ConvBNReLU(out_ch, out_ch, stride=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class ScratchED(nn.Module):
    def __init__(self, in_channels: int = 3, base_ch: int = 32):
        super().__init__()
        c1, c2, c3, c4 = base_ch, base_ch * 2, base_ch * 4, base_ch * 8

        # Encoder (no maxpool; stride=2 on first conv of stage)
        self.enc1 = DoubleConv(in_channels, c1, stride=1)
        self.enc2 = DoubleConv(c1, c2, stride=2)
        self.enc3 = DoubleConv(c2, c3, stride=2)

        # Bottleneck
        self.bot = DoubleConv(c3, c4, stride=2)

        # Decoder (no skip connections)
        self.dec3 = DoubleConv(c4, c3, stride=1)
        self.dec2 = DoubleConv(c3, c2, stride=1)
        self.dec1 = DoubleConv(c2, c1, stride=1)

        # Head
        self.out_conv = nn.Conv2d(c1, 1, kernel_size=1)

    def _upsample(self, x: torch.Tensor, ref: torch.Tensor) -> torch.Tensor:
        return F.interpolate(x, size=ref.shape[-2:], mode="bilinear", align_corners=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Encoder
        e1 = self.enc1(x)          # H,   W
        e2 = self.enc2(e1)         # H/2, W/2
        e3 = self.enc3(e2)         # H/4, W/4

        b = self.bot(e3)           # H/8, W/8

        # Decoder (progressively upsample to previous encoder scales, but no skips)
        d3 = self._upsample(self.dec3(b), e3)  # -> H/4, W/4
        d2 = self._upsample(self.dec2(d3), e2) # -> H/2, W/2
        d1 = self._upsample(self.dec1(d2), e1) # -> H,   W

        return self.out_conv(d1)


__all__ = ["ScratchED"]

