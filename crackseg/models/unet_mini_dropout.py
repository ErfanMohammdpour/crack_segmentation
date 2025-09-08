from __future__ import annotations

"""Lightweight U-Net-like architecture with Dropout2d blocks."""

import torch
import torch.nn as nn


class ConvBlockDrop(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, p: float = 0.3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Dropout2d(p=p),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Dropout2d(p=p),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class UNetMiniDropout(nn.Module):
    def __init__(self, in_channels: int = 3, base_ch: int = 32, dropout: float = 0.3):
        super().__init__()
        c1, c2, c3, c4 = base_ch, base_ch * 2, base_ch * 4, base_ch * 8

        self.enc1 = ConvBlockDrop(in_channels, c1, dropout)
        self.pool1 = nn.MaxPool2d(2)
        self.enc2 = ConvBlockDrop(c1, c2, dropout)
        self.pool2 = nn.MaxPool2d(2)
        self.enc3 = ConvBlockDrop(c2, c3, dropout)
        self.pool3 = nn.MaxPool2d(2)

        self.bottleneck = ConvBlockDrop(c3, c4, dropout)

        self.up3 = nn.ConvTranspose2d(c4, c3, kernel_size=2, stride=2)
        self.dec3 = ConvBlockDrop(c4, c3, dropout)
        self.up2 = nn.ConvTranspose2d(c3, c2, kernel_size=2, stride=2)
        self.dec2 = ConvBlockDrop(c3, c2, dropout)
        self.up1 = nn.ConvTranspose2d(c2, c1, kernel_size=2, stride=2)
        self.dec1 = ConvBlockDrop(c2, c1, dropout)

        self.out_conv = nn.Conv2d(c1, 1, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool1(e1))
        e3 = self.enc3(self.pool2(e2))

        b = self.bottleneck(self.pool3(e3))

        d3 = self.up3(b)
        d3 = torch.cat([d3, e3], dim=1)
        d3 = self.dec3(d3)
        d2 = self.up2(d3)
        d2 = torch.cat([d2, e2], dim=1)
        d2 = self.dec2(d2)
        d1 = self.up1(d2)
        d1 = torch.cat([d1, e1], dim=1)
        d1 = self.dec1(d1)

        return self.out_conv(d1)

