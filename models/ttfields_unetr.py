from __future__ import annotations

from typing import Dict

import torch
import torch.nn as nn

from monai.networks.nets import UNETR


class TTFieldsUNETR(nn.Module):
    """
    UNETR-based model for BraTS tumor segmentation.

    Phase 1: segmentation only — coord outputs are zero placeholders.
    Phase 2: CoordinateHead will be attached to the ViT bottleneck.

    Output contract matches TinyUNet3D:
        seg_logits : (B, out_channels, D, H, W)
        coord_wt   : (B, 3)  — zeros in phase 1
        coord_et   : (B, 3)  — zeros in phase 1
    """

    def __init__(
        self,
        in_channels: int = 4,
        out_channels: int = 4,
        img_size: int = 128,
        feature_size: int = 16,
        hidden_size: int = 768,
        mlp_dim: int = 3072,
        num_heads: int = 12,
        dropout_rate: float = 0.0,
    ) -> None:
        super().__init__()

        self.in_channels = in_channels
        self.img_size = img_size

        self.unetr = UNETR(
            in_channels=in_channels,
            out_channels=out_channels,
            img_size=(img_size, img_size, img_size),
            feature_size=feature_size,
            hidden_size=hidden_size,
            mlp_dim=mlp_dim,
            num_heads=num_heads,
            dropout_rate=dropout_rate,
            spatial_dims=3,
        )

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        if x.ndim != 5:
            raise ValueError(
                f"Expected input shape (B,C,D,H,W), got {tuple(x.shape)}"
            )

        if x.shape[1] != self.in_channels:
            raise ValueError(
                f"Expected {self.in_channels} input channels, got {x.shape[1]}"
            )

        expected = (self.img_size,) * 3
        if tuple(x.shape[2:]) != expected:
            raise ValueError(
                f"Expected spatial size {expected}, got {tuple(x.shape[2:])}"
            )

        seg_logits = self.unetr(x)

        batch_size = x.shape[0]
        zeros = x.new_zeros((batch_size, 3))

        return {
            "seg_logits": seg_logits,
            "coord_wt": zeros,
            "coord_et": zeros,
        }
