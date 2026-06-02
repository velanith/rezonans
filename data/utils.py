from __future__ import annotations

from typing import Dict, Tuple, Optional

import torch
import numpy as np


BRATS_LABELS = {
    "background": 0,
    "ncr": 1,
    "ed": 2,
    "et": 4,
}

REMAPPED_LABELS = {
    "background": 0,
    "ncr": 1,
    "ed": 2,
    "et": 3,
}


def remap_brats_labels(mask: torch.Tensor) -> torch.Tensor:
    """
    Remap BraTS labels from:
        0 = background
        1 = NCR
        2 = ED
        4 = ET

    to contiguous training labels:
        0 = background
        1 = NCR
        2 = ED
        3 = ET

    Args:
        mask: Tensor with shape (D, H, W), values in {0,1,2,4}

    Returns:
        Tensor with shape (D, H, W), values in {0,1,2,3}
    """
    if mask.ndim != 3:
        raise ValueError(f"Expected mask shape (D,H,W), got {tuple(mask.shape)}")

    remapped = mask.clone()
    remapped[remapped == 4] = 3

    valid = torch.isin(
        remapped,
        torch.tensor([0, 1, 2, 3], device=remapped.device),
    )

    if not bool(valid.all()):
        unique = torch.unique(mask).detach().cpu().tolist()
        raise ValueError(f"Unexpected BraTS labels found: {unique}")

    return remapped.long()


def to_regions(mask: torch.Tensor) -> Dict[str, torch.Tensor]:
    """
    Convert remapped BraTS class mask into WT / TC / ET regions.

    Input labels:
        0 = background
        1 = NCR
        2 = ED
        3 = ET

    Regions:
        WT = NCR + ED + ET
        TC = NCR + ET
        ET = ET

    Args:
        mask:
            Tensor with shape (D, H, W), values in {0,1,2,3}

    Returns:
        {
            "wt": Tensor (D,H,W),
            "tc": Tensor (D,H,W),
            "et": Tensor (D,H,W),
        }
    """
    if mask.ndim != 3:
        raise ValueError(f"Expected mask shape (D,H,W), got {tuple(mask.shape)}")

    wt = mask > 0
    tc = (mask == 1) | (mask == 3)
    et = mask == 3

    return {
        "wt": wt.long(),
        "tc": tc.long(),
        "et": et.long(),
    }


def zyx_to_xyz(
    coord_zyx: torch.Tensor | np.ndarray,
) -> torch.Tensor | np.ndarray:
    """
    Convert coordinate order from (z, y, x) to (x, y, z).

    Args:
        coord_zyx: Shape (..., 3)

    Returns:
        Coordinate reordered as (..., x, y, z)
    """
    if coord_zyx.shape[-1] != 3:
        raise ValueError(
            f"Expected last dimension to be 3, got {coord_zyx.shape[-1]}"
        )

    return coord_zyx[..., [2, 1, 0]]


def xyz_to_zyx(
    coord_xyz: torch.Tensor | np.ndarray,
) -> torch.Tensor | np.ndarray:
    """
    Convert coordinate order from (x, y, z) to (z, y, x).

    Args:
        coord_xyz: Shape (..., 3)

    Returns:
        Coordinate reordered as (..., z, y, x)
    """
    if coord_xyz.shape[-1] != 3:
        raise ValueError(
            f"Expected last dimension to be 3, got {coord_xyz.shape[-1]}"
        )

    return coord_xyz[..., [2, 1, 0]]


def apply_affine_zyx(
    affine: np.ndarray | torch.Tensor,
    coord_zyx: np.ndarray | torch.Tensor,
) -> np.ndarray | torch.Tensor:
    """
    Apply voxel-to-world affine to coordinates stored in internal (z,y,x) order.

    Important:
        Internal coordinate order is (z,y,x).
        Affine expects voxel coordinates in (x,y,z) order.
        Output world coordinates are returned in (x_mm,y_mm,z_mm).

    Args:
        affine:
            Shape (4,4)

        coord_zyx:
            Shape (...,3), voxel coordinates in (z,y,x)

    Returns:
        World coordinates in (x_mm,y_mm,z_mm), shape (...,3)
    """
    if affine.shape != (4, 4):
        raise ValueError(
            f"Expected affine shape (4,4), got {tuple(affine.shape)}"
        )

    if coord_zyx.shape[-1] != 3:
        raise ValueError(
            f"Expected coord last dimension 3, got {coord_zyx.shape[-1]}"
        )

    coord_xyz = zyx_to_xyz(coord_zyx)

    if isinstance(coord_xyz, torch.Tensor):
        aff = affine

        if not isinstance(aff, torch.Tensor):
            aff = torch.as_tensor(
                aff,
                dtype=coord_xyz.dtype,
                device=coord_xyz.device,
            )
        else:
            aff = aff.to(
                dtype=coord_xyz.dtype,
                device=coord_xyz.device,
            )

        ones = torch.ones(
            (*coord_xyz.shape[:-1], 1),
            dtype=coord_xyz.dtype,
            device=coord_xyz.device,
        )

        hom = torch.cat([coord_xyz, ones], dim=-1)

        world = hom @ aff.T
        return world[..., :3]

    aff_np = np.asarray(affine, dtype=coord_xyz.dtype)

    ones = np.ones(
        (*coord_xyz.shape[:-1], 1),
        dtype=coord_xyz.dtype,
    )

    hom = np.concatenate([coord_xyz, ones], axis=-1)

    world = hom @ aff_np.T
    return world[..., :3]


def normalize_coord(
    coord_zyx: torch.Tensor,
    shape_zyx: Tuple[int, int, int],
) -> torch.Tensor:
    """
    Normalize voxel coordinates from voxel space to [-1, 1].

    Formula:
        v_norm = 2 * v / (dim - 1) - 1

    Args:
        coord_zyx:
            Tensor with shape (..., 3)
            Coordinate order: (z, y, x)

        shape_zyx:
            Volume shape as (D, H, W)

    Returns:
        Normalized coordinates with shape (..., 3)
    """
    if coord_zyx.shape[-1] != 3:
        raise ValueError(
            f"Expected coord last dimension 3, got {coord_zyx.shape[-1]}"
        )

    shape = torch.tensor(
        shape_zyx,
        dtype=coord_zyx.dtype,
        device=coord_zyx.device,
    )

    return 2.0 * coord_zyx / (shape - 1.0) - 1.0


def denormalize_coord(
    coord_norm_zyx: torch.Tensor,
    shape_zyx: Tuple[int, int, int],
) -> torch.Tensor:
    """
    Convert normalized [-1,1] coordinates back to voxel space.

    Formula:
        v_voxel = (v_norm + 1) / 2 * (dim - 1)

    Args:
        coord_norm_zyx:
            Tensor with shape (..., 3)
            Coordinate order: (z, y, x)

        shape_zyx:
            Volume shape as (D, H, W)

    Returns:
        Voxel coordinates in (z, y, x)
    """
    if coord_norm_zyx.shape[-1] != 3:
        raise ValueError(
            f"Expected coord last dimension 3, got {coord_norm_zyx.shape[-1]}"
        )

    shape = torch.tensor(
        shape_zyx,
        dtype=coord_norm_zyx.dtype,
        device=coord_norm_zyx.device,
    )

    return (coord_norm_zyx + 1.0) / 2.0 * (shape - 1.0)


def _voxel_centroid(
    binary_mask: torch.Tensor,
) -> Optional[torch.Tensor]:
    """
    Compute centroid of a binary mask in voxel coordinates.

    Args:
        binary_mask:
            Tensor with shape (D, H, W)

    Returns:
        Tensor with shape (3,) in (z,y,x),
        or None if mask is empty.
    """
    if binary_mask.ndim != 3:
        raise ValueError(
            f"Expected binary_mask shape (D,H,W), got {tuple(binary_mask.shape)}"
        )

    coords = torch.nonzero(binary_mask, as_tuple=False).float()

    if coords.shape[0] == 0:
        return None

    return coords.mean(dim=0)


def compute_centroid_with_fallback(
    mask: torch.Tensor,
    shape_zyx: Tuple[int, int, int] | None = None,
) -> Dict[str, torch.Tensor]:
    """
    Compute WT and ET centroids from a remapped BraTS mask.

    Input mask labels:
        0 = background
        1 = NCR
        2 = ED
        3 = ET

    WT centroid:
        Always computed from mask > 0.
        If WT is empty, fallback is volume center.

    ET centroid:
        Computed from mask == 3 if ET exists.
        If ET does not exist, fallback is WT centroid.
        Loss function should mask ET loss using has_et=False.

    Args:
        mask:
            Tensor with shape (D, H, W), values in {0,1,2,3}

        shape_zyx:
            Shape used for normalization.
            If None, uses mask.shape.

    Returns:
        {
            "wt_centroid_norm": Tensor (3,),
            "et_centroid_norm": Tensor (3,),
            "wt_centroid_voxel": Tensor (3,),
            "et_centroid_voxel": Tensor (3,),
            "has_et": Tensor bool scalar,
        }
    """
    if mask.ndim != 3:
        raise ValueError(
            f"Expected mask shape (D,H,W), got {tuple(mask.shape)}"
        )

    if shape_zyx is None:
        final_shape: Tuple[int, int, int] = (
            mask.shape[0],
            mask.shape[1],
            mask.shape[2],
        )
    else:
        final_shape = shape_zyx

    wt_mask = mask > 0
    et_mask = mask == 3

    wt_voxel = _voxel_centroid(wt_mask)

    if wt_voxel is None:
        d, h, w = final_shape

        wt_voxel = torch.tensor(
            [
                (d - 1) / 2.0,
                (h - 1) / 2.0,
                (w - 1) / 2.0,
            ],
            dtype=torch.float32,
            device=mask.device,
        )

    et_voxel = _voxel_centroid(et_mask)
    has_et = et_voxel is not None

    if et_voxel is None:
        et_voxel = wt_voxel.clone()

    wt_norm = normalize_coord(wt_voxel, final_shape)
    et_norm = normalize_coord(et_voxel, final_shape)

    return {
        "wt_centroid_norm": wt_norm,
        "et_centroid_norm": et_norm,
        "wt_centroid_voxel": wt_voxel,
        "et_centroid_voxel": et_voxel,
        "has_et": torch.tensor(
            has_et,
            dtype=torch.bool,
            device=mask.device,
        ),
    }

def bbox_from_mask(
    mask: torch.Tensor,
) -> Optional[Tuple[int, int, int, int, int, int]]:
    """
    Compute bounding box from a binary or label mask.

    Args:
        mask:
            Tensor with shape (D, H, W).
            Nonzero voxels are treated as foreground.

    Returns:
        Bounding box in zyx order:
            (z0, y0, x0, z1, y1, x1)

        z1, y1, x1 are exclusive.

        Returns None if mask is empty.
    """
    if mask.ndim != 3:
        raise ValueError(
            f"Expected mask shape (D,H,W), got {tuple(mask.shape)}"
        )

    coords = torch.nonzero(mask > 0, as_tuple=False)

    if coords.shape[0] == 0:
        return None

    mins = coords.min(dim=0).values
    maxs = coords.max(dim=0).values + 1

    z0, y0, x0 = [int(v) for v in mins.tolist()]
    z1, y1, x1 = [int(v) for v in maxs.tolist()]

    return z0, y0, x0, z1, y1, x1

def tumor_volume_mm3(
    mask: torch.Tensor,
    affine: np.ndarray | torch.Tensor,
) -> float:
    """
    Compute tumor volume in mm^3 from a binary or label mask.

    Args:
        mask:
            Tensor with shape (D, H, W).
            Nonzero voxels are treated as tumor.

        affine:
            Voxel-to-world affine matrix with shape (4, 4).

    Returns:
        Tumor volume in mm^3.
    """
    if mask.ndim != 3:
        raise ValueError(
            f"Expected mask shape (D,H,W), got {tuple(mask.shape)}"
        )

    if affine.shape != (4, 4):
        raise ValueError(
            f"Expected affine shape (4,4), got {tuple(affine.shape)}"
        )

    num_voxels = int((mask > 0).sum().item())

    if isinstance(affine, torch.Tensor):
        affine_np = affine.detach().cpu().numpy()
    else:
        affine_np = affine

    voxel_volume = abs(float(np.linalg.det(affine_np[:3, :3])))

    return num_voxels * voxel_volume