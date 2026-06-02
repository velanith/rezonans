from pathlib import Path

import pytest
import torch

from data.dataset import (
    BraTSDataset,
    load_brats_case,
    zscore_normalize,
)


CASE_ROOT = Path("/Users/yemre/Desktop/rezonans/dataset")
CASE_ID = "BraTS2021_01163"


def case_available() -> bool:
    case_dir = CASE_ROOT / CASE_ID
    return case_dir.exists()


@pytest.mark.skipif(
    not case_available(),
    reason="Local BraTS test case is not available",
)
def test_load_brats_case_shapes_and_labels():
    case = load_brats_case(CASE_ROOT / CASE_ID)

    image = case["image"]
    mask = case["mask"]
    affine = case["affine"]

    assert isinstance(image, torch.Tensor)
    assert isinstance(mask, torch.Tensor)

    assert image.shape == (4, 155, 240, 240)
    assert mask.shape == (155, 240, 240)
    assert affine.shape == (4, 4)

    assert image.dtype == torch.float32
    assert mask.dtype == torch.long

    labels = set(torch.unique(mask).tolist())
    assert labels.issubset({0, 1, 2, 3})
    assert labels == {0, 1, 2, 3}


@pytest.mark.skipif(
    not case_available(),
    reason="Local BraTS test case is not available",
)
def test_zscore_normalize_preserves_shape_and_is_finite():
    case = load_brats_case(CASE_ROOT / CASE_ID)

    image = case["image"]
    image_norm = zscore_normalize(image)

    assert image_norm.shape == image.shape
    assert image_norm.dtype == torch.float32
    assert torch.isfinite(image_norm).all()


@pytest.mark.skipif(
    not case_available(),
    reason="Local BraTS test case is not available",
)
def test_zscore_normalize_brain_region_roughly_standardized():
    case = load_brats_case(CASE_ROOT / CASE_ID)

    image = case["image"]
    image_norm = zscore_normalize(image)

    brain_mask = image.abs().sum(dim=0) > 0

    for c in range(image.shape[0]):
        voxels = image_norm[c][brain_mask]

        assert abs(float(voxels.mean())) < 1e-4
        assert abs(float(voxels.std()) - 1.0) < 1e-4


@pytest.mark.skipif(
    not case_available(),
    reason="Local BraTS test case is not available",
)
def test_brats_dataset_getitem():
    dataset = BraTSDataset(
        root_dir=CASE_ROOT,
        case_ids=[CASE_ID],
        crop_size=128,
        jitter=32,
        normalize=True,
    )

    sample = dataset[0]

    assert sample["image"].shape == (4, 128, 128, 128)
    assert sample["seg"].shape == (128, 128, 128)

    assert sample["image"].dtype == torch.float32
    assert sample["seg"].dtype == torch.long

    assert sample["wt_centroid_norm"].shape == (3,)
    assert sample["et_centroid_norm"].shape == (3,)

    assert torch.all(sample["wt_centroid_norm"] >= -1.0)
    assert torch.all(sample["wt_centroid_norm"] <= 1.0)
    assert torch.all(sample["et_centroid_norm"] >= -1.0)
    assert torch.all(sample["et_centroid_norm"] <= 1.0)

    assert sample["has_et"].dtype == torch.bool
    assert sample["crop_origin_zyx"].shape == (3,)
    assert sample["orig_shape_zyx"].shape == (3,)

    assert sample["orig_shape_zyx"].tolist() == [155, 240, 240]
    assert sample["meta"]["case_id"] == CASE_ID
    assert sample["meta"]["affine"].shape == (4, 4)


@pytest.mark.skipif(
    not case_available(),
    reason="Local BraTS test case is not available",
)
def test_brats_dataset_crop_contains_foreground():
    dataset = BraTSDataset(
        root_dir=CASE_ROOT,
        case_ids=[CASE_ID],
        crop_size=128,
        jitter=32,
        normalize=True,
    )

    sample = dataset[0]

    assert int((sample["seg"] > 0).sum().item()) > 0


def test_brats_dataset_empty_case_ids_raises():
    with pytest.raises(ValueError):
        BraTSDataset(
            root_dir=CASE_ROOT,
            case_ids=[],
        )


def test_brats_dataset_missing_root_raises():
    with pytest.raises(FileNotFoundError):
        BraTSDataset(
            root_dir="/definitely/not/a/real/path",
            case_ids=[CASE_ID],
        )