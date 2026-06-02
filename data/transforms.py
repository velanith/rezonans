from __future__ import annotations

from typing import Tuple, TypedDict

import torch
import torch.nn.functional as F


class ForegroundCropResult(TypedDict):
    image: torch.Tensor
    mask: torch.Tensor
    crop_origin_zyx: Tuple[int, int, int]


def compute_crop_origin(
    center_zyx: Tuple[int, int, int],
    crop_size: int,
    full_shape_zyx: Tuple[int, int, int],
) -> Tuple[int, int, int]:
    """
    Compute crop origin for a fixed-size 3D crop.

    The crop is centered around center_zyx as much as possible,
    but shifted inward if it would exceed volume boundaries.
    """
    if crop_size <= 0:
        raise ValueError(f"crop_size must be positive, got {crop_size}")

    origins: list[int] = []

    for center, dim in zip(center_zyx, full_shape_zyx):
        if dim <= 0:
            raise ValueError(
                f"Volume dimensions must be positive, got {full_shape_zyx}"
            )

        half = crop_size // 2

        if dim >= crop_size:
            origin = center - half
            origin = max(0, min(origin, dim - crop_size))
        else:
            origin = 0

        origins.append(origin)

    return origins[0], origins[1], origins[2]


def crop_or_pad_3d(
    volume: torch.Tensor,
    origin_zyx: Tuple[int, int, int],
    crop_size: int,
    pad_value: float = 0.0,
) -> torch.Tensor:
    """
    Crop a 3D tensor to fixed size, padding if needed.

    Args:
        volume: Tensor with shape (..., D, H, W).
        origin_zyx: Crop origin in (z,y,x).
        crop_size: Desired output size.
        pad_value: Constant padding value.

    Returns:
        Tensor with shape (..., crop_size, crop_size, crop_size).
    """
    if volume.ndim < 3:
        raise ValueError(
            f"Expected volume with at least 3 dims (...,D,H,W), got {tuple(volume.shape)}"
        )

    if crop_size <= 0:
        raise ValueError(f"crop_size must be positive, got {crop_size}")

    oz, oy, ox = origin_zyx

    if oz < 0 or oy < 0 or ox < 0:
        raise ValueError(f"origin_zyx must be non-negative, got {origin_zyx}")

    crop = volume[
        ...,
        oz : oz + crop_size,
        oy : oy + crop_size,
        ox : ox + crop_size,
    ]

    dz = crop_size - crop.shape[-3]
    dy = crop_size - crop.shape[-2]
    dx = crop_size - crop.shape[-1]

    if dz > 0 or dy > 0 or dx > 0:
        crop = F.pad(
            crop,
            pad=(
                0,
                max(dx, 0),
                0,
                max(dy, 0),
                0,
                max(dz, 0),
            ),
            mode="constant",
            value=pad_value,
        )

    if crop.shape[-3:] != (crop_size, crop_size, crop_size):
        raise RuntimeError(
            "crop_or_pad_3d failed to produce expected shape. "
            f"Expected {crop_size}³, got {tuple(crop.shape[-3:])}"
        )

    return crop


def sample_foreground_crop_center(
    mask: torch.Tensor,
    jitter: int = 0,
) -> Tuple[int, int, int]:
    """
    Sample a random foreground voxel from a mask, optionally with jitter.

    Args:
        mask: Tensor with shape (D,H,W).
        jitter: Random offset range. If 32, offset is sampled from [-32, 32].

    Returns:
        Crop center as (z,y,x).
    """
    if mask.ndim != 3:
        raise ValueError(
            f"Expected mask shape (D,H,W), got {tuple(mask.shape)}"
        )

    if jitter < 0:
        raise ValueError(f"jitter must be non-negative, got {jitter}")

    d, h, w = mask.shape
    fg_coords = torch.nonzero(mask > 0, as_tuple=False)

    if fg_coords.shape[0] > 0:
        idx = int(torch.randint(
            low=0,
            high=fg_coords.shape[0],
            size=(1,),
            device=fg_coords.device,
        ).item())

        center = fg_coords[idx].clone()

        if jitter > 0:
            offset = torch.randint(
                low=-jitter,
                high=jitter + 1,
                size=(3,),
                device=center.device,
            )
            center = center + offset

        center[0] = center[0].clamp(0, d - 1)
        center[1] = center[1].clamp(0, h - 1)
        center[2] = center[2].clamp(0, w - 1)

        return (
            int(center[0].item()),
            int(center[1].item()),
            int(center[2].item()),
        )

    return (
        int(torch.randint(0, d, (1,)).item()),
        int(torch.randint(0, h, (1,)).item()),
        int(torch.randint(0, w, (1,)).item()),
    )


def foreground_crop(
    image: torch.Tensor,
    mask: torch.Tensor,
    crop_size: int = 128,
    jitter: int = 0,
) -> ForegroundCropResult:
    """
    Create a foreground-aware crop from image and mask.

    Args:
        image: Tensor with shape (C,D,H,W).
        mask: Tensor with shape (D,H,W).
        crop_size: Output crop size.
        jitter: Random offset range applied to sampled foreground center.

    Returns:
        {
            "image": Tensor (C,crop_size,crop_size,crop_size),
            "mask": Tensor (crop_size,crop_size,crop_size),
            "crop_origin_zyx": Tuple[int, int, int],
        }
    """
    if image.ndim != 4:
        raise ValueError(
            f"Expected image shape (C,D,H,W), got {tuple(image.shape)}"
        )

    if mask.ndim != 3:
        raise ValueError(
            f"Expected mask shape (D,H,W), got {tuple(mask.shape)}"
        )

    if image.shape[-3:] != mask.shape:
        raise ValueError(
            f"Image spatial shape {tuple(image.shape[-3:])} "
            f"does not match mask shape {tuple(mask.shape)}"
        )

    center_zyx = sample_foreground_crop_center(mask, jitter=jitter)

    full_shape_zyx: Tuple[int, int, int] = (
        image.shape[-3],
        image.shape[-2],
        image.shape[-1],
    )

    origin_zyx = compute_crop_origin(
        center_zyx=center_zyx,
        crop_size=crop_size,
        full_shape_zyx=full_shape_zyx,
    )

    image_crop = crop_or_pad_3d(
        image,
        origin_zyx=origin_zyx,
        crop_size=crop_size,
        pad_value=0.0,
    )

    mask_crop = crop_or_pad_3d(
        mask,
        origin_zyx=origin_zyx,
        crop_size=crop_size,
        pad_value=0.0,
    ).long()

    return {
        "image": image_crop,
        "mask": mask_crop,
        "crop_origin_zyx": origin_zyx,
    }