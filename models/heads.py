from __future__ import annotations

import torch
import torch.nn as nn


class CoordinateHead(nn.Module):
    """
    Soft-argmax coordinate regression head.

    Predicts WT and ET centroids using separate spatial attention maps
    over the ViT patch grid (8x8x8), then refines with an MLP.

    Coordinate order: (z, y, x), output range [-1, 1].
    """

    def __init__(
        self,
        in_channels: int,
        hidden_dim: int = 256,
        use_tanh: bool = True,
    ) -> None:
        super().__init__()

        if in_channels <= 0:
            raise ValueError(f"in_channels must be positive, got {in_channels}")
        if hidden_dim <= 0:
            raise ValueError(f"hidden_dim must be positive, got {hidden_dim}")

        self.in_channels = in_channels
        self.hidden_dim = hidden_dim
        self.use_tanh = use_tanh

        # Separate 1x1 conv attention maps for WT and ET
        self.attn_wt = nn.Conv3d(in_channels, 1, kernel_size=1)
        self.attn_et = nn.Conv3d(in_channels, 1, kernel_size=1)

        # Refinement MLPs: attend-pooled features + soft-argmax coord → delta
        self.refine_wt = nn.Sequential(
            nn.Linear(in_channels + 3, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Linear(hidden_dim // 2, 3),
        )
        self.refine_et = nn.Sequential(
            nn.Linear(in_channels + 3, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Linear(hidden_dim // 2, 3),
        )

    @staticmethod
    def _coord_grid(D: int, H: int, W: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        zz = torch.linspace(-1, 1, D, device=device, dtype=dtype)
        yy = torch.linspace(-1, 1, H, device=device, dtype=dtype)
        xx = torch.linspace(-1, 1, W, device=device, dtype=dtype)
        gz, gy, gx = torch.meshgrid(zz, yy, xx, indexing="ij")
        return torch.stack([gz, gy, gx], dim=0)  # (3, D, H, W)

    def _soft_argmax(
        self,
        x: torch.Tensor,
        attn_conv: nn.Conv3d,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        B, C, D, H, W = x.shape
        attn = attn_conv(x).view(B, -1).softmax(-1).view(B, 1, D, H, W)
        grid = self._coord_grid(D, H, W, x.device, x.dtype)   # (3, D, H, W)
        coord = (attn * grid.unsqueeze(0)).sum(dim=[2, 3, 4])  # (B, 3)
        feat  = (attn * x).sum(dim=[2, 3, 4])                  # (B, C)
        return coord, feat

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        """
        Args:
            x: (B, C, D, H, W) — ViT patch grid reshaped to 3-D

        Returns:
            {"coord_wt": (B, 3), "coord_et": (B, 3)}
        """
        if x.ndim != 5:
            raise ValueError(f"Expected input shape (B,C,D,H,W), got {tuple(x.shape)}")
        if x.shape[1] != self.in_channels:
            raise ValueError(f"Expected {self.in_channels} input channels, got {x.shape[1]}")

        coord_wt, feat_wt = self._soft_argmax(x, self.attn_wt)
        coord_et, feat_et = self._soft_argmax(x, self.attn_et)

        # Residual refinement
        coord_wt = coord_wt + self.refine_wt(torch.cat([feat_wt, coord_wt], dim=-1))
        coord_et = coord_et + self.refine_et(torch.cat([feat_et, coord_et], dim=-1))

        if self.use_tanh:
            coord_wt = coord_wt.tanh()
            coord_et = coord_et.tanh()

        return {"coord_wt": coord_wt, "coord_et": coord_et}
