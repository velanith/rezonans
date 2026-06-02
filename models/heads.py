from __future__ import annotations

import torch
import torch.nn as nn


class CoordinateHead(nn.Module):
    """
    Shared coordinate regression head.

    Predicts:
        - WT centroid: 3 values
        - ET centroid: 3 values

    Coordinate order:
        (z, y, x)

    Output range:
        [-1, 1] because centroids are crop-local normalized.
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

        layers: list[nn.Module] = [
            nn.AdaptiveAvgPool3d(1),
            nn.Flatten(),
            nn.Linear(in_channels, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Linear(hidden_dim // 2, 6),
        ]

        if use_tanh:
            layers.append(nn.Tanh())

        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        """
        Args:
            x:
                Feature tensor with shape (B, C, D, H, W)

        Returns:
            {
                "coord_wt": Tensor (B, 3),
                "coord_et": Tensor (B, 3),
            }
        """
        if x.ndim != 5:
            raise ValueError(
                f"Expected input shape (B,C,D,H,W), got {tuple(x.shape)}"
            )

        if x.shape[1] != self.in_channels:
            raise ValueError(
                f"Expected {self.in_channels} input channels, got {x.shape[1]}"
            )

        out = self.net(x)

        return {
            "coord_wt": out[:, 0:3],
            "coord_et": out[:, 3:6],
        }