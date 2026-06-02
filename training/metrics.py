from __future__ import annotations

from typing import Dict, Tuple

import numpy as np
import torch

from data.utils import apply_affine_zyx, denormalize_coord, to_regions


def dice_score_binary(
    pred: torch.Tensor,
    target: torch.Tensor,
    eps: float = 1e-8,
) -> torch.Tensor:
    """
    Compute binary Dice score.

    Args:
        pred: Binary prediction tensor.
        target: Binary target tensor.
        eps: Numerical stability constant.

    Returns:
        Scalar Dice score.
    """
    if pred.shape != target.shape:
        raise ValueError(
            f"Shape mismatch: pred {tuple(pred.shape)}, target {tuple(target.shape)}"
        )

    pred = pred.bool()
    target = target.bool()

    intersection = torch.logical_and(pred, target).sum().float()
    denominator = pred.sum().float() + target.sum().float()

    if denominator.item() == 0:
        return torch.tensor(1.0, device=pred.device)

    return (2.0 * intersection + eps) / (denominator + eps)


def dice_wt_tc_et(
    pred_mask: torch.Tensor,
    target_mask: torch.Tensor,
) -> Dict[str, torch.Tensor]:
    """
    Compute WT / TC / ET Dice scores from remapped BraTS masks.

    Args:
        pred_mask:
            Tensor with shape (D,H,W), labels 0,1,2,3.

        target_mask:
            Tensor with shape (D,H,W), labels 0,1,2,3.

    Returns:
        {
            "dice_wt": scalar tensor,
            "dice_tc": scalar tensor,
            "dice_et": scalar tensor,
        }
    """
    if pred_mask.ndim != 3:
        raise ValueError(
            f"Expected pred_mask shape (D,H,W), got {tuple(pred_mask.shape)}"
        )

    if target_mask.ndim != 3:
        raise ValueError(
            f"Expected target_mask shape (D,H,W), got {tuple(target_mask.shape)}"
        )

    if pred_mask.shape != target_mask.shape:
        raise ValueError(
            f"Shape mismatch: pred {tuple(pred_mask.shape)}, "
            f"target {tuple(target_mask.shape)}"
        )

    pred_regions = to_regions(pred_mask)
    target_regions = to_regions(target_mask)

    return {
        "dice_wt": dice_score_binary(pred_regions["wt"], target_regions["wt"]),
        "dice_tc": dice_score_binary(pred_regions["tc"], target_regions["tc"]),
        "dice_et": dice_score_binary(pred_regions["et"], target_regions["et"]),
    }


def coord_mae_norm(
    pred_norm_zyx: torch.Tensor,
    target_norm_zyx: torch.Tensor,
) -> torch.Tensor:
    """
    Compute coordinate MAE in normalized [-1,1] space.

    Args:
        pred_norm_zyx:
            Tensor with shape (...,3).

        target_norm_zyx:
            Tensor with shape (...,3).

    Returns:
        Scalar MAE.
    """
    if pred_norm_zyx.shape != target_norm_zyx.shape:
        raise ValueError(
            f"Shape mismatch: pred {tuple(pred_norm_zyx.shape)}, "
            f"target {tuple(target_norm_zyx.shape)}"
        )

    if pred_norm_zyx.shape[-1] != 3:
        raise ValueError(
            f"Expected last dimension 3, got {pred_norm_zyx.shape[-1]}"
        )

    return torch.mean(torch.abs(pred_norm_zyx - target_norm_zyx))


def coord_mae_mm(
    pred_norm_zyx: torch.Tensor,
    target_norm_zyx: torch.Tensor,
    crop_origin_zyx: torch.Tensor,
    affine: np.ndarray | torch.Tensor,
    crop_shape_zyx: Tuple[int, int, int] = (128, 128, 128),
) -> torch.Tensor:
    """
    Compute Euclidean coordinate error in world-space millimeters.

    Args:
        pred_norm_zyx:
            Tensor with shape (B,3), normalized crop-local coordinates.

        target_norm_zyx:
            Tensor with shape (B,3), normalized crop-local coordinates.

        crop_origin_zyx:
            Tensor with shape (B,3), crop origins in full-volume voxel space.

        affine:
            Affine matrix with shape (4,4).
            For now this function supports one affine shared by the batch.

        crop_shape_zyx:
            Crop shape used for denormalization.

    Returns:
        Scalar mean Euclidean distance in mm.
    """
    if pred_norm_zyx.shape != target_norm_zyx.shape:
        raise ValueError(
            f"Shape mismatch: pred {tuple(pred_norm_zyx.shape)}, "
            f"target {tuple(target_norm_zyx.shape)}"
        )

    if pred_norm_zyx.ndim != 2 or pred_norm_zyx.shape[-1] != 3:
        raise ValueError(
            f"Expected pred_norm_zyx shape (B,3), got {tuple(pred_norm_zyx.shape)}"
        )

    if crop_origin_zyx.shape != pred_norm_zyx.shape:
        raise ValueError(
            f"Expected crop_origin_zyx shape {tuple(pred_norm_zyx.shape)}, "
            f"got {tuple(crop_origin_zyx.shape)}"
        )

    pred_crop_voxel = denormalize_coord(pred_norm_zyx, crop_shape_zyx)
    target_crop_voxel = denormalize_coord(target_norm_zyx, crop_shape_zyx)

    origin = crop_origin_zyx.to(
        device=pred_norm_zyx.device,
        dtype=pred_norm_zyx.dtype,
    )

    pred_full_voxel = pred_crop_voxel + origin
    target_full_voxel = target_crop_voxel + origin

    pred_world = apply_affine_zyx(affine, pred_full_voxel)
    target_world = apply_affine_zyx(affine, target_full_voxel)

    if not isinstance(pred_world, torch.Tensor):
        pred_world = torch.as_tensor(
            pred_world,
            dtype=pred_norm_zyx.dtype,
            device=pred_norm_zyx.device,
        )

    if not isinstance(target_world, torch.Tensor):
        target_world = torch.as_tensor(
            target_world,
            dtype=target_norm_zyx.dtype,
            device=target_norm_zyx.device,
        )

    errors = torch.linalg.norm(pred_world - target_world, dim=-1)

    return errors.mean()