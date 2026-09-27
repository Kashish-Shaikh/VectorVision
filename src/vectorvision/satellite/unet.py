"""U-Net for 64 x 64 Sentinel-2 tiles. Randomly initialised: no pretrained weights."""
from __future__ import annotations

import torch
import torch.nn as nn


def block(i: int, o: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv2d(i, o, 3, padding=1, bias=False), nn.BatchNorm2d(o), nn.ReLU(inplace=True),
        nn.Conv2d(o, o, 3, padding=1, bias=False), nn.BatchNorm2d(o), nn.ReLU(inplace=True))


class UNet(nn.Module):
    """Three down-sampling levels: 64 -> 32 -> 16 -> 8.

    A fourth level would shrink to 4 x 4, where the receptive field is already larger
    than the whole tile and the skip connections stop carrying useful detail.
    """

    def __init__(self, n_classes: int, width: int = 32, in_ch: int = 3):
        super().__init__()
        w = width
        self.e1, self.e2, self.e3 = block(in_ch, w), block(w, 2 * w), block(2 * w, 4 * w)
        self.mid = block(4 * w, 8 * w)
        self.u3, self.d3 = nn.ConvTranspose2d(8 * w, 4 * w, 2, 2), block(8 * w, 4 * w)
        self.u2, self.d2 = nn.ConvTranspose2d(4 * w, 2 * w, 2, 2), block(4 * w, 2 * w)
        self.u1, self.d1 = nn.ConvTranspose2d(2 * w, w, 2, 2), block(2 * w, w)
        self.head = nn.Conv2d(w, n_classes, 1)
        self.pool = nn.MaxPool2d(2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        e1 = self.e1(x)
        e2 = self.e2(self.pool(e1))
        e3 = self.e3(self.pool(e2))
        m = self.mid(self.pool(e3))
        d3 = self.d3(torch.cat([self.u3(m), e3], 1))
        d2 = self.d2(torch.cat([self.u2(d3), e2], 1))
        d1 = self.d1(torch.cat([self.u1(d2), e1], 1))
        return self.head(d1)


def count_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
