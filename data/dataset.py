from __future__ import annotations

from pathlib import Path
from typing import Dict, Tuple, TypedDict

import nibabel as nib
import numpy as np
import torch

from data.transforms import foreground_crop
from data.utils import (
    apply_affine_zyx,
    compute_centroid_with_fallback,
    remap_brats_labels,
)


MODALITIES = ("t1", "t1ce", "t2", "flair")


class SingleCaseOutput(TypedDict):
    image: torch.Tensor
    mask: torch.Tensor
    affine: np.ndarray
    case_id: str


def load_nifti(path: str | Path) -> tuple[np.ndarray, np.ndarray]:
    """
    Load a NIfTI file.

    Returns:
        data:
            Numpy array as loaded by nibabel.

        affine:
            Voxel-to-world affine matrix.
    """
    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(f"NIfTI file not found: {path}")

    nii = nib.load(str(path))
    data = nii.get_fdata(dtype=np.float32)
    affine = nii.affine.astype(np.float32)

    return data, affine


def find_case_file(case_dir: str | Path, case_id: str, suffix: str) -> Path:
    """
    Find a BraTS file inside a case directory.

    Example:
        BraTS2021_01163_t1.nii.gz
        BraTS2021_01163_seg.nii.gz
    """
    case_dir = Path(case_dir)
    path = case_dir / f"{case_id}_{suffix}.nii.gz"

    if not path.exists():
        raise FileNotFoundError(f"Expected file not found: {path}")

    return path


def load_brats_case(case_dir: str | Path) -> SingleCaseOutput:
    """
    Load one BraTS case.

    Expected files:
        {case_id}_t1.nii.gz
        {case_id}_t1ce.nii.gz
        {case_id}_t2.nii.gz
        {case_id}_flair.nii.gz
        {case_id}_seg.nii.gz

    Returns:
        image:
            Tensor with shape (4,D,H,W)

        mask:
            Tensor with shape (D,H,W), remapped labels 0,1,2,3

        affine:
            Affine matrix from image file.

        case_id:
            Case folder name.
    """
    case_dir = Path(case_dir)
    case_id = case_dir.name

    volumes = []
    affine_ref: np.ndarray | None = None
    shape_ref: tuple[int, int, int] | None = None

    for modality in MODALITIES:
        path = find_case_file(case_dir, case_id, modality)
        data, affine = load_nifti(path)

        # nibabel usually loads BraTS as (H,W,D) = (240,240,155)
        # internal convention is (D,H,W), so transpose.
        data_dhw = np.transpose(data, (2, 0, 1))

        if shape_ref is None:
            shape_ref = data_dhw.shape
            affine_ref = affine
        else:
            if data_dhw.shape != shape_ref:
                raise ValueError(
                    f"Shape mismatch for {path.name}: "
                    f"got {data_dhw.shape}, expected {shape_ref}"
                )

        volumes.append(torch.from_numpy(data_dhw).float())

    image = torch.stack(volumes, dim=0)

    seg_path = find_case_file(case_dir, case_id, "seg")
    seg_data, seg_affine = load_nifti(seg_path)
    seg_dhw = np.transpose(seg_data, (2, 0, 1))

    if shape_ref is not None and seg_dhw.shape != shape_ref:
        raise ValueError(
            f"Segmentation shape mismatch: got {seg_dhw.shape}, expected {shape_ref}"
        )

    mask_raw = torch.from_numpy(seg_dhw).long()
    mask = remap_brats_labels(mask_raw)

    if affine_ref is None:
        raise RuntimeError("affine_ref was not initialized")

    return {
        "image": image,
        "mask": mask,
        "affine": affine_ref,
        "case_id": case_id,
    }


def zscore_normalize(
    image: torch.Tensor,
    eps: float = 1e-8,
) -> torch.Tensor:
    """
    Z-score normalize each modality inside the brain mask.

    Args:
        image:
            Tensor with shape (C,D,H,W)

    Returns:
        Normalized tensor with same shape.
    """
    if image.ndim != 4:
        raise ValueError(f"Expected image shape (C,D,H,W), got {tuple(image.shape)}")

    brain_mask = image.abs().sum(dim=0) > 0

    out = image.clone()

    for c in range(image.shape[0]):
        voxels = image[c][brain_mask]

        if voxels.numel() == 0:
            continue

        mean = voxels.mean()
        std = voxels.std().clamp_min(eps)

        out[c] = (image[c] - mean) / std

    return out


def debug_single_case(case_dir: str | Path, crop_size: int = 128) -> None:
    """
    Run a smoke test on a single BraTS case.
    """
    case = load_brats_case(case_dir)

    image = case["image"]
    mask = case["mask"]
    affine = case["affine"]
    case_id = case["case_id"]

    print(f"case_id: {case_id}")
    print(f"image shape: {tuple(image.shape)}")
    print(f"mask shape : {tuple(mask.shape)}")
    print(f"labels     : {torch.unique(mask).tolist()}")
    print(f"affine     :\n{affine}")

    image_norm = zscore_normalize(image)

    crop = foreground_crop(
        image=image_norm,
        mask=mask,
        crop_size=crop_size,
        jitter=32,
    )

    image_crop = crop["image"]
    mask_crop = crop["mask"]
    crop_origin_zyx = crop["crop_origin_zyx"]

    centroids = compute_centroid_with_fallback(
        mask_crop,
        shape_zyx=(crop_size, crop_size, crop_size),
    )

    wt_crop_voxel = centroids["wt_centroid_voxel"]
    et_crop_voxel = centroids["et_centroid_voxel"]

    origin = torch.tensor(crop_origin_zyx, dtype=torch.float32)

    wt_full_voxel = wt_crop_voxel + origin
    et_full_voxel = et_crop_voxel + origin

    wt_world = apply_affine_zyx(affine, wt_full_voxel)
    et_world = apply_affine_zyx(affine, et_full_voxel)

    print("\n--- crop ---")
    print(f"image_crop shape: {tuple(image_crop.shape)}")
    print(f"mask_crop shape : {tuple(mask_crop.shape)}")
    print(f"crop_origin_zyx : {crop_origin_zyx}")
    print(f"crop labels      : {torch.unique(mask_crop).tolist()}")

    print("\n--- centroids ---")
    print(f"has_et: {bool(centroids['has_et'].item())}")
    print(f"WT crop voxel zyx : {wt_crop_voxel.tolist()}")
    print(f"WT full voxel zyx : {wt_full_voxel.tolist()}")
    print(f"WT world xyz mm   : {wt_world.tolist()}")
    print(f"ET crop voxel zyx : {et_crop_voxel.tolist()}")
    print(f"ET full voxel zyx : {et_full_voxel.tolist()}")
    print(f"ET world xyz mm   : {et_world.tolist()}")

from torch.utils.data import Dataset

class BraTSDataset(Dataset[dict]):
    def __init__(
        self,
        root_dir: str | Path,
        case_ids: list[str],
        crop_size: int = 128,
        jitter: int = 32,
        normalize: bool = True,
    ) -> None:
        self.root_dir = Path(root_dir)
        self.case_ids = case_ids
        self.crop_size = crop_size
        self.jitter = jitter
        self.normalize = normalize

        if not self.root_dir.exists():
            raise FileNotFoundError(f"Dataset root not found: {self.root_dir}")

        if len(self.case_ids) == 0:
            raise ValueError("case_ids cannot be empty")

    def __len__(self) -> int:
        return len(self.case_ids)

    def __getitem__(self, idx: int) -> dict:
        case_id = self.case_ids[idx]
        case_dir = self.root_dir / case_id

        case = load_brats_case(case_dir)

        image = case["image"]
        mask = case["mask"]
        affine = case["affine"]

        orig_shape_zyx = (
            image.shape[-3],
            image.shape[-2],
            image.shape[-1],
        )

        if self.normalize:
            image = zscore_normalize(image)

        crop = foreground_crop(
            image=image,
            mask=mask,
            crop_size=self.crop_size,
            jitter=self.jitter,
        )

        image_crop = crop["image"]
        mask_crop = crop["mask"]
        crop_origin_zyx = crop["crop_origin_zyx"]

        centroids = compute_centroid_with_fallback(
            mask_crop,
            shape_zyx=(self.crop_size, self.crop_size, self.crop_size),
        )

        return {
            "image": image_crop.float(),
            "seg": mask_crop.long(),

            "wt_centroid_norm": centroids["wt_centroid_norm"].float(),
            "et_centroid_norm": centroids["et_centroid_norm"].float(),
            "has_et": centroids["has_et"],

            "crop_origin_zyx": torch.tensor(
                crop_origin_zyx,
                dtype=torch.long,
            ),

            "orig_shape_zyx": torch.tensor(
                orig_shape_zyx,
                dtype=torch.long,
            ),

            "meta": {
                "case_id": case_id,
                "affine": affine,
            },
        }