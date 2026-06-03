from __future__ import annotations

from typing import Dict

import torch
import torch.nn as nn

from monai.networks.nets import UNETR

from models.heads import CoordinateHead


class TTFieldsUNETR(nn.Module):
    """
    UNETR-based model for BraTS tumor segmentation and centroid localization.

    Phase 1 (use_coord_head=False):
        seg_logits from UNETR, coord outputs are zero placeholders.

    Phase 2 (use_coord_head=True):
        ViT final hidden state is captured via forward hook, reshaped to
        (B, hidden_size, 8, 8, 8), and passed through CoordinateHead.

    Output contract:
        seg_logits : (B, out_channels, D, H, W)
        coord_wt   : (B, 3)
        coord_et   : (B, 3)
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
        use_coord_head: bool = False,
    ) -> None:
        super().__init__()

        self.in_channels = in_channels
        self.img_size = img_size
        self.hidden_size = hidden_size
        self.use_coord_head = use_coord_head

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

        if use_coord_head:
            self.coord_head = CoordinateHead(in_channels=hidden_size)
            self._bottleneck: torch.Tensor | None = None
            self.unetr.vit.register_forward_hook(self._capture_vit_output)
        else:
            self.coord_head = None

    def _capture_vit_output(
        self,
        module: nn.Module,
        input: tuple,
        output: tuple,
    ) -> None:
        # ViT returns (final_hidden_state, hidden_states_list)
        # final_hidden_state: (B, N_patches, hidden_size)
        self._bottleneck = output[0]

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

        if self.use_coord_head and self._bottleneck is not None:
            B = x.shape[0]
            # (B, N_patches, hidden_size) → (B, hidden_size, P, P, P)
            patch_dim = round(self._bottleneck.shape[1] ** (1 / 3))
            bottleneck_3d = (
                self._bottleneck
                .transpose(1, 2)
                .reshape(B, self.hidden_size, patch_dim, patch_dim, patch_dim)
            )
            self._bottleneck = None

            coords = self.coord_head(bottleneck_3d)
            coord_wt = coords["coord_wt"]
            coord_et = coords["coord_et"]
        else:
            zeros = x.new_zeros((x.shape[0], 3))
            coord_wt = zeros
            coord_et = zeros

        return {
            "seg_logits": seg_logits,
            "coord_wt": coord_wt,
            "coord_et": coord_et,
        }
