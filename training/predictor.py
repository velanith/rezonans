from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from data.dataset import load_brats_case, zscore_normalize
from data.transforms import compute_crop_origin, crop_or_pad_3d
from data.utils import apply_affine_zyx, denormalize_coord
from training.utils import autocast_context


@dataclass
class CasePrediction:
    case_id: str
    seg_full: torch.Tensor          # (D, H, W), labels 0-3, full volume
    wt_centroid_voxel_zyx: list[float]
    et_centroid_voxel_zyx: list[float]
    wt_centroid_mm: list[float]     # world space (x, y, z) mm
    et_centroid_mm: list[float]
    wt_voxel_count: int             # predicted WT voxels in crop
    mean_wt_confidence: float       # mean tumor probability over predicted WT region
    review_recommended: bool        # True if model confidence is low


def _brain_center_zyx(image: torch.Tensor) -> tuple[int, int, int]:
    """Centroid of non-zero voxels across all channels — deterministic crop center."""
    brain_mask = image.abs().sum(dim=0) > 0
    coords = torch.nonzero(brain_mask, as_tuple=False).float()
    if coords.shape[0] == 0:
        d, h, w = image.shape[-3:]
        return (d // 2, h // 2, w // 2)
    mean = coords.mean(dim=0)
    return (int(mean[0].item()), int(mean[1].item()), int(mean[2].item()))


def _reconstruct_seg(
    pred_crop: torch.Tensor,
    crop_origin_zyx: tuple[int, int, int],
    full_shape_zyx: tuple[int, int, int],
    crop_size: int,
) -> torch.Tensor:
    """Place crop-space segmentation back into the full volume."""
    full_seg = torch.zeros(full_shape_zyx, dtype=torch.long)
    oz, oy, ox = crop_origin_zyx
    ez = min(oz + crop_size, full_shape_zyx[0])
    ey = min(oy + crop_size, full_shape_zyx[1])
    ex = min(ox + crop_size, full_shape_zyx[2])
    full_seg[oz:ez, oy:ey, ox:ex] = pred_crop[: ez - oz, : ey - oy, : ex - ox]
    return full_seg


def _coord_to_world(
    coord_norm: torch.Tensor,
    crop_origin_zyx: tuple[int, int, int],
    crop_size: int,
    affine: np.ndarray,
) -> list[float]:
    """Normalized crop-local coord → world mm."""
    crop_shape = (crop_size,) * 3
    voxel_crop = denormalize_coord(coord_norm.unsqueeze(0), crop_shape).squeeze(0)
    origin = torch.tensor(crop_origin_zyx, dtype=torch.float32)
    voxel_full = voxel_crop + origin
    world = apply_affine_zyx(affine, voxel_full.unsqueeze(0)).squeeze(0)
    return [round(float(v), 3) for v in world.tolist()]


@torch.no_grad()
def predict_case(
    model: nn.Module,
    case_dir: Path,
    device: torch.device,
    precision: str = "fp32",
    crop_size: int = 128,
) -> CasePrediction:
    case = load_brats_case(case_dir)
    image = zscore_normalize(case["image"])
    affine = case["affine"]
    case_id = case["case_id"]

    full_shape_zyx = (image.shape[-3], image.shape[-2], image.shape[-1])

    center_zyx = _brain_center_zyx(image)
    origin_zyx = compute_crop_origin(center_zyx, crop_size, full_shape_zyx)

    image_crop = crop_or_pad_3d(image, origin_zyx, crop_size)

    model.eval()
    inp = image_crop.unsqueeze(0).to(device)

    with autocast_context(device, precision):
        model_out = model(inp)

    seg_logits = model_out["seg_logits"]
    softmax_probs = torch.softmax(seg_logits, dim=1).squeeze(0).cpu()  # (4, D, H, W)
    pred_crop = torch.argmax(seg_logits, dim=1).squeeze(0).cpu()
    seg_full = _reconstruct_seg(pred_crop, origin_zyx, full_shape_zyx, crop_size)

    # Confidence: tumor probability = 1 - P(background)
    tumor_prob = 1.0 - softmax_probs[0]  # (D, H, W)
    wt_mask = pred_crop > 0
    wt_voxel_count = int(wt_mask.sum().item())
    if wt_voxel_count > 0:
        mean_wt_confidence = float(tumor_prob[wt_mask].mean().item())
    else:
        mean_wt_confidence = 0.0
    review_recommended = wt_voxel_count < 500 or mean_wt_confidence < 0.5

    coord_wt = model_out["coord_wt"].squeeze(0).cpu()
    coord_et = model_out["coord_et"].squeeze(0).cpu()

    origin_t = torch.tensor(origin_zyx, dtype=torch.float32)
    wt_crop_vox = denormalize_coord(coord_wt.unsqueeze(0), (crop_size,) * 3).squeeze(0)
    et_crop_vox = denormalize_coord(coord_et.unsqueeze(0), (crop_size,) * 3).squeeze(0)
    wt_full_vox = (wt_crop_vox + origin_t).tolist()
    et_full_vox = (et_crop_vox + origin_t).tolist()

    wt_mm = _coord_to_world(coord_wt, origin_zyx, crop_size, affine)
    et_mm = _coord_to_world(coord_et, origin_zyx, crop_size, affine)

    return CasePrediction(
        case_id=case_id,
        seg_full=seg_full,
        wt_centroid_voxel_zyx=[round(v, 2) for v in wt_full_vox],
        et_centroid_voxel_zyx=[round(v, 2) for v in et_full_vox],
        wt_centroid_mm=wt_mm,
        et_centroid_mm=et_mm,
        wt_voxel_count=wt_voxel_count,
        mean_wt_confidence=round(mean_wt_confidence, 4),
        review_recommended=review_recommended,
    )
