from __future__ import annotations

from typing import Dict

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.heads import CoordinateHead


class ConvBlock3D(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
    ) -> None:
        super().__init__()

        self.net = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.InstanceNorm3d(out_channels),
            nn.GELU(),
            nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.InstanceNorm3d(out_channels),
            nn.GELU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class TinyUNet3D(nn.Module):
    """
    Tiny 3D U-Net for pipeline validation.

    This is not the final model.
    It exists to verify:
        - dataset
        - loss
        - metrics
        - training loop
        - coordinate head

    Output contract matches TTFieldsUNETR.
    """

    def __init__(
        self,
        in_channels: int = 4,
        out_channels: int = 4,
        base_channels: int = 8,
        use_coord_head: bool = True,
    ) -> None:
        super().__init__()

        if in_channels <= 0:
            raise ValueError(f"in_channels must be positive, got {in_channels}")

        if out_channels <= 0:
            raise ValueError(f"out_channels must be positive, got {out_channels}")

        if base_channels <= 0:
            raise ValueError(f"base_channels must be positive, got {base_channels}")

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.base_channels = base_channels
        self.use_coord_head = use_coord_head

        c = base_channels

        self.enc1 = ConvBlock3D(in_channels, c)
        self.enc2 = ConvBlock3D(c, c * 2)
        self.enc3 = ConvBlock3D(c * 2, c * 4)

        self.pool = nn.MaxPool3d(kernel_size=2, stride=2)

        self.bottleneck = ConvBlock3D(c * 4, c * 8)

        self.up3 = nn.ConvTranspose3d(c * 8, c * 4, kernel_size=2, stride=2)
        self.dec3 = ConvBlock3D(c * 8, c * 4)

        self.up2 = nn.ConvTranspose3d(c * 4, c * 2, kernel_size=2, stride=2)
        self.dec2 = ConvBlock3D(c * 4, c * 2)

        self.up1 = nn.ConvTranspose3d(c * 2, c, kernel_size=2, stride=2)
        self.dec1 = ConvBlock3D(c * 2, c)

        self.seg_head = nn.Conv3d(c, out_channels, kernel_size=1)

        if use_coord_head:
            self.coord_head = CoordinateHead(in_channels=c * 8)
        else:
            self.coord_head = None

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        if x.ndim != 5:
            raise ValueError(
                f"Expected input shape (B,C,D,H,W), got {tuple(x.shape)}"
            )

        if x.shape[1] != self.in_channels:
            raise ValueError(
                f"Expected {self.in_channels} input channels, got {x.shape[1]}"
            )

        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))

        b = self.bottleneck(self.pool(e3))

        d3 = self.up3(b)
        d3 = self._match_and_concat(d3, e3)
        d3 = self.dec3(d3)

        d2 = self.up2(d3)
        d2 = self._match_and_concat(d2, e2)
        d2 = self.dec2(d2)

        d1 = self.up1(d2)
        d1 = self._match_and_concat(d1, e1)
        d1 = self.dec1(d1)

        seg_logits = self.seg_head(d1)

        if self.coord_head is not None:
            coords = self.coord_head(b)
            coord_wt = coords["coord_wt"]
            coord_et = coords["coord_et"]
        else:
            batch_size = x.shape[0]
            coord_wt = x.new_zeros((batch_size, 3))
            coord_et = x.new_zeros((batch_size, 3))

        return {
            "seg_logits": seg_logits,
            "coord_wt": coord_wt,
            "coord_et": coord_et,
        }

    @staticmethod
    def _match_and_concat(
        x: torch.Tensor,
        skip: torch.Tensor,
    ) -> torch.Tensor:
        """
        Match x spatial shape to skip if odd input sizes cause mismatch,
        then concatenate along channel dimension.
        """
        if x.shape[-3:] != skip.shape[-3:]:
            x = F.interpolate(
                x,
                size=skip.shape[-3:],
                mode="trilinear",
                align_corners=False,
            )

        return torch.cat([x, skip], dim=1)