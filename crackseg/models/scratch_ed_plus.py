from __future__ import annotations

"""Stronger from-scratch encoder–decoder without U-Net skips.

Enhancements over ScratchED:
- Residual blocks within stages (same-scale residuals only).
- Squeeze-and-Excitation (SE) channel attention after stage outputs.
- ASPP at bottleneck with configurable dilation rates, fused with 1×1 conv.
"""

from typing import List
import torch
import torch.nn as nn
import torch.nn.functional as F


def conv3x3(in_ch: int, out_ch: int, stride: int = 1) -> nn.Conv2d:
    return nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=stride, padding=1, bias=False)

def _maybe_dropout(p: float) -> nn.Module:
    """Return Identity if p<=0 else Dropout2d(p).

    Dropout is active only during training (PyTorch default), and has no
    learnable parameters, keeping checkpoints backward-compatible.
    """
    return nn.Identity() if p <= 0.0 else nn.Dropout2d(p)

class SEBlock(nn.Module):
    def __init__(self, ch: int, reduction: int = 16):
        super().__init__()
        hidden = max(8, ch // reduction)
        self.fc1 = nn.Conv2d(ch, hidden, kernel_size=1)
        self.fc2 = nn.Conv2d(hidden, ch, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        s = x.mean(dim=(2, 3), keepdim=True)
        s = F.relu(self.fc1(s), inplace=True)
        s = torch.sigmoid(self.fc2(s))
        return x * s


class ResidualBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, stride: int = 1, se: bool = True):
        super().__init__()
        self.conv1 = conv3x3(in_ch, out_ch, stride)
        self.bn1 = nn.BatchNorm2d(out_ch)
        self.conv2 = conv3x3(out_ch, out_ch, 1)
        self.bn2 = nn.BatchNorm2d(out_ch)
        self.act = nn.ReLU(inplace=True)
        self.se = SEBlock(out_ch) if se else nn.Identity()
        self.down = None
        if in_ch != out_ch or stride != 1:
            self.down = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_ch),
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x
        out = self.act(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        if self.down is not None:
            identity = self.down(identity)
        out = self.se(self.act(out + identity))
        return out


class ASPP(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, rates: List[int]):
        super().__init__()
        branches = []
        for r in rates:
            if r == 1:
                branches.append(nn.Conv2d(in_ch, out_ch, kernel_size=1, bias=False))
            else:
                branches.append(
                    nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=r, dilation=r, bias=False)
                )
        self.branches = nn.ModuleList(branches)
        self.bn = nn.BatchNorm2d(out_ch * len(branches))
        self.act = nn.ReLU(inplace=True)
        self.fuse = nn.Conv2d(out_ch * len(branches), out_ch, kernel_size=1, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        outs = [b(x) for b in self.branches]
        x = torch.cat(outs, dim=1)
        x = self.act(self.bn(x))
        return self.fuse(x)


class ScratchEDPlus(nn.Module):
    def __init__(
        self,
        in_channels: int = 3,
        base_ch: int = 32,
        aspp_rates: List[int] | None = None,
        dropout_p: float = 0.0,
    ):
        super().__init__()
        if aspp_rates is None:
            aspp_rates = [1, 6, 12, 18]
        # store for reference; modules use fixed instances created below
        self.dropout_p = float(dropout_p)
        c1, c2, c3, c4 = base_ch, base_ch * 2, base_ch * 4, base_ch * 8

        # Encoder (stride-2 at stage entry)
        self.enc1 = nn.Sequential(
            ResidualBlock(in_channels, c1, stride=1, se=True),
            ResidualBlock(c1, c1, stride=1, se=True),
        )
        self.enc2 = nn.Sequential(
            ResidualBlock(c1, c2, stride=2, se=True),
            ResidualBlock(c2, c2, stride=1, se=True),
            _maybe_dropout(self.dropout_p),  # dropout in deeper encoder stage
        )
        self.enc3 = nn.Sequential(
            ResidualBlock(c2, c3, stride=2, se=True),
            ResidualBlock(c3, c3, stride=1, se=True),
            _maybe_dropout(self.dropout_p),  # dropout in deeper encoder stage
        )

        # Bottleneck with ASPP
        self.bot_pre = ResidualBlock(c3, c4, stride=2, se=True)  # H/8
        self.aspp = ASPP(c4, c4, rates=aspp_rates)
        # Dropout after ASPP fusion acts as bottleneck regularization
        self.drop_aspp = _maybe_dropout(self.dropout_p)

        # Decoder (no skips)
        self.dec3 = nn.Sequential(
            ResidualBlock(c4, c3, stride=1, se=True),
            ResidualBlock(c3, c3, stride=1, se=True),
            _maybe_dropout(self.dropout_p),  # deeper decoder stage
        )
        self.dec2 = nn.Sequential(
            ResidualBlock(c3, c2, stride=1, se=True),
            ResidualBlock(c2, c2, stride=1, se=True),
            _maybe_dropout(self.dropout_p),  # deeper decoder stage
        )
        self.dec1 = nn.Sequential(
            ResidualBlock(c2, c1, stride=1, se=True),
            ResidualBlock(c1, c1, stride=1, se=True),
        )

        self.head = nn.Conv2d(c1, 1, kernel_size=1)

    def _upsample(self, x: torch.Tensor, ref: torch.Tensor) -> torch.Tensor:
        return F.interpolate(x, size=ref.shape[-2:], mode="bilinear", align_corners=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        e1 = self.enc1(x)          # H,   W
        e2 = self.enc2(e1)         # H/2, W/2
        e3 = self.enc3(e2)         # H/4, W/4

        b = self.bot_pre(e3)       # H/8, W/8
        b = self.aspp(b)
        b = self.drop_aspp(b)

        d3 = self._upsample(self.dec3(b), e3)  # -> H/4
        d2 = self._upsample(self.dec2(d3), e2) # -> H/2
        d1 = self._upsample(self.dec1(d2), e1) # -> H

        return self.head(d1)


__all__ = ["ScratchEDPlus"]
