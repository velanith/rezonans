import numpy as np
import torch
import pytest

from data.utils import (
    remap_brats_labels,
    to_regions,
    zyx_to_xyz,
    xyz_to_zyx,
    apply_affine_zyx,
    normalize_coord,
    denormalize_coord,
    compute_centroid_with_fallback,
    bbox_from_mask,
    tumor_volume_mm3,
)


def test_remap_brats_labels():
    mask = torch.tensor(
        [
            [[0, 1], [2, 4]],
        ]
    )

    out = remap_brats_labels(mask)

    expected = torch.tensor(
        [
            [[0, 1], [2, 3]],
        ]
    )

    assert torch.equal(out, expected)
    assert out.dtype == torch.long


def test_to_regions():
    mask = torch.tensor(
        [
            [[0, 1], [2, 3]],
        ]
    )

    regions = to_regions(mask)

    expected_wt = torch.tensor([[[0, 1], [1, 1]]])
    expected_tc = torch.tensor([[[0, 1], [0, 1]]])
    expected_et = torch.tensor([[[0, 0], [0, 1]]])

    assert torch.equal(regions["wt"], expected_wt)
    assert torch.equal(regions["tc"], expected_tc)
    assert torch.equal(regions["et"], expected_et)


def test_axis_conversion_roundtrip():
    coord_zyx = torch.tensor([10.0, 20.0, 30.0])

    coord_xyz = zyx_to_xyz(coord_zyx)
    assert isinstance(coord_xyz, torch.Tensor)

    roundtrip = xyz_to_zyx(coord_xyz)
    assert isinstance(roundtrip, torch.Tensor)

    assert torch.equal(coord_xyz, torch.tensor([30.0, 20.0, 10.0]))
    assert torch.equal(roundtrip, coord_zyx)


def test_apply_affine_zyx_identity():
    affine = np.eye(4, dtype=np.float32)
    coord_zyx = torch.tensor([10.0, 20.0, 30.0])

    world_xyz = apply_affine_zyx(affine, coord_zyx)
    assert isinstance(world_xyz, torch.Tensor)

    assert torch.allclose(world_xyz, torch.tensor([30.0, 20.0, 10.0]))

def test_apply_affine_zyx_translation():
    affine = np.eye(4, dtype=np.float32)
    affine[:3, 3] = np.array([100.0, 200.0, 300.0], dtype=np.float32)

    coord_zyx = torch.tensor([10.0, 20.0, 30.0])

    world_xyz = apply_affine_zyx(affine, coord_zyx)
    assert isinstance(world_xyz, torch.Tensor)

    expected = torch.tensor([130.0, 220.0, 310.0])
    assert torch.allclose(world_xyz, expected)


def test_normalize_denormalize_roundtrip():
    shape = (128, 128, 128)
    coord = torch.tensor([0.0, 63.5, 127.0])

    norm = normalize_coord(coord, shape)
    recovered = denormalize_coord(norm, shape)

    assert torch.allclose(recovered, coord, atol=1e-5)


def test_compute_centroid_with_fallback_with_et():
    mask = torch.zeros((8, 8, 8), dtype=torch.long)

    mask[2, 2, 2] = 1
    mask[4, 4, 4] = 3
    mask[6, 6, 6] = 2

    out = compute_centroid_with_fallback(mask)

    expected_wt_voxel = torch.tensor([4.0, 4.0, 4.0])
    expected_et_voxel = torch.tensor([4.0, 4.0, 4.0])

    assert torch.allclose(out["wt_centroid_voxel"], expected_wt_voxel)
    assert torch.allclose(out["et_centroid_voxel"], expected_et_voxel)
    assert bool(out["has_et"].item()) is True


def test_compute_centroid_with_fallback_without_et():
    mask = torch.zeros((8, 8, 8), dtype=torch.long)

    mask[2, 2, 2] = 1
    mask[6, 6, 6] = 2

    out = compute_centroid_with_fallback(mask)

    expected_wt_voxel = torch.tensor([4.0, 4.0, 4.0])

    assert torch.allclose(out["wt_centroid_voxel"], expected_wt_voxel)
    assert torch.allclose(out["et_centroid_voxel"], expected_wt_voxel)
    assert bool(out["has_et"].item()) is False


def test_bbox_from_mask():
    mask = torch.zeros((10, 20, 30), dtype=torch.long)

    mask[2:5, 4:8, 10:15] = 1

    bbox = bbox_from_mask(mask)

    assert bbox == (2, 4, 10, 5, 8, 15)


def test_bbox_from_empty_mask():
    mask = torch.zeros((10, 20, 30), dtype=torch.long)

    bbox = bbox_from_mask(mask)

    assert bbox is None


def test_tumor_volume_mm3_identity_affine():
    mask = torch.zeros((4, 4, 4), dtype=torch.long)
    mask[0:2, 0:2, 0:2] = 1

    affine = np.eye(4, dtype=np.float32)

    volume = tumor_volume_mm3(mask, affine)

    assert volume == 8.0


def test_tumor_volume_mm3_scaled_affine():
    mask = torch.zeros((4, 4, 4), dtype=torch.long)
    mask[0:2, 0:2, 0:2] = 1

    affine = np.eye(4, dtype=np.float32)
    affine[0, 0] = 2.0
    affine[1, 1] = 3.0
    affine[2, 2] = 4.0

    volume = tumor_volume_mm3(mask, affine)

    assert volume == 8 * 2 * 3 * 4

# ─── remap_brats_labels ───────────────────────────────────────────────────────

def test_remap_et_4_to_3():
    mask = torch.tensor([[[0, 1, 2, 4]]], dtype=torch.long)
    result = remap_brats_labels(mask)
    assert result.tolist() == [[[0, 1, 2, 3]]]

def test_remap_does_not_modify_input():
    mask = torch.tensor([[[0, 1, 2, 4]]], dtype=torch.long)
    remap_brats_labels(mask)
    assert mask[0, 0, 3].item() == 4  # in-place değil

def test_remap_output_dtype_is_long():
    mask = torch.tensor([[[0, 1, 2, 4]]], dtype=torch.long)
    result = remap_brats_labels(mask)
    assert result.dtype == torch.long

def test_remap_rejects_2d_input():
    mask = torch.tensor([[0, 1, 2, 4]], dtype=torch.long)
    with pytest.raises(ValueError):
        remap_brats_labels(mask)

def test_remap_rejects_invalid_labels():
    mask = torch.tensor([[[0, 1, 2, 5]]], dtype=torch.long)
    with pytest.raises(ValueError):
        remap_brats_labels(mask)


# ─── to_regions ───────────────────────────────────────────────────────────────

def make_mask():
    # 4x4x4, basit dağılım
    mask = torch.zeros(4, 4, 4, dtype=torch.long)
    mask[0, 0, 0] = 1  # NCR
    mask[1, 1, 1] = 2  # ED
    mask[2, 2, 2] = 3  # ET
    return mask

def test_wt_includes_all_tumor():
    regions = to_regions(make_mask())
    assert regions["wt"][0, 0, 0].item() == 1
    assert regions["wt"][1, 1, 1].item() == 1
    assert regions["wt"][2, 2, 2].item() == 1
    assert regions["wt"][3, 3, 3].item() == 0  # background

def test_tc_includes_ncr_and_et():
    regions = to_regions(make_mask())
    assert regions["tc"][0, 0, 0].item() == 1  # NCR
    assert regions["tc"][2, 2, 2].item() == 1  # ET
    assert regions["tc"][1, 1, 1].item() == 0  # ED — TC'de yok

def test_et_only_et():
    regions = to_regions(make_mask())
    assert regions["et"][2, 2, 2].item() == 1
    assert regions["et"][0, 0, 0].item() == 0
    assert regions["et"][1, 1, 1].item() == 0

def test_regions_are_long():
    regions = to_regions(make_mask())
    for v in regions.values():
        assert v.dtype == torch.long


# ─── zyx_to_xyz / xyz_to_zyx ──────────────────────────────────────────────────

def test_zyx_to_xyz_reorders():
    coord = torch.tensor([1.0, 2.0, 3.0])
    result = zyx_to_xyz(coord)
    assert result.tolist() == [3.0, 2.0, 1.0]

def test_xyz_to_zyx_reorders():
    coord = torch.tensor([3.0, 2.0, 1.0])
    result = xyz_to_zyx(coord)
    assert result.tolist() == [1.0, 2.0, 3.0]

def test_zyx_xyz_roundtrip():
    coord = torch.tensor([5.0, 10.0, 20.0])
    assert xyz_to_zyx(zyx_to_xyz(coord)).tolist() == coord.tolist()

def test_zyx_to_xyz_batched():
    coords = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    result = zyx_to_xyz(coords)
    assert result[0].tolist() == [3.0, 2.0, 1.0]
    assert result[1].tolist() == [6.0, 5.0, 4.0]


# ─── normalize_coord / denormalize_coord ──────────────────────────────────────

def test_normalize_center():
    # Merkez voxel → 0.0
    shape = (128, 128, 128)
    coord = torch.tensor([63.5, 63.5, 63.5])
    result = normalize_coord(coord, shape)
    assert torch.allclose(result, torch.zeros(3), atol=1e-5)

def test_normalize_origin():
    # (0,0,0) → (-1,-1,-1)
    shape = (128, 128, 128)
    coord = torch.zeros(3)
    result = normalize_coord(coord, shape)
    assert torch.allclose(result, -torch.ones(3), atol=1e-5)

def test_normalize_max():
    # (127,127,127) → (1,1,1)
    shape = (128, 128, 128)
    coord = torch.tensor([127.0, 127.0, 127.0])
    result = normalize_coord(coord, shape)
    assert torch.allclose(result, torch.ones(3), atol=1e-5)

def test_normalize_denormalize_roundtrip():
    shape = (128, 128, 128)
    coord = torch.tensor([30.0, 60.0, 90.0])
    assert torch.allclose(
        denormalize_coord(normalize_coord(coord, shape), shape),
        coord,
        atol=1e-4,
    )


# ─── compute_centroid_with_fallback ───────────────────────────────────────────

def test_centroid_wt_correct():
    mask = torch.zeros(10, 10, 10, dtype=torch.long)
    mask[5, 5, 5] = 1
    result = compute_centroid_with_fallback(mask)
    assert torch.allclose(result["wt_centroid_voxel"],
                          torch.tensor([5.0, 5.0, 5.0]), atol=1e-4)

def test_centroid_has_et_true():
    mask = torch.zeros(10, 10, 10, dtype=torch.long)
    mask[5, 5, 5] = 3
    result = compute_centroid_with_fallback(mask)
    assert result["has_et"].item() is True

def test_centroid_has_et_false():
    mask = torch.zeros(10, 10, 10, dtype=torch.long)
    mask[5, 5, 5] = 1  # sadece NCR, ET yok
    result = compute_centroid_with_fallback(mask)
    assert result["has_et"].item() is False

def test_centroid_et_fallback_to_wt():
    mask = torch.zeros(10, 10, 10, dtype=torch.long)
    mask[5, 5, 5] = 1  # ET yok
    result = compute_centroid_with_fallback(mask)
    assert torch.allclose(
        result["et_centroid_voxel"],
        result["wt_centroid_voxel"],
        atol=1e-4,
    )

def test_centroid_empty_mask_fallback():
    mask = torch.zeros(10, 10, 10, dtype=torch.long)
    result = compute_centroid_with_fallback(mask)
    # Volume center olmalı
    expected = torch.tensor([4.5, 4.5, 4.5])
    assert torch.allclose(result["wt_centroid_voxel"], expected, atol=1e-4)

def test_centroid_norm_in_range():
    mask = torch.zeros(128, 128, 128, dtype=torch.long)
    mask[10, 20, 30] = 2
    result = compute_centroid_with_fallback(mask)
    assert result["wt_centroid_norm"].abs().max().item() <= 1.0


# ─── bbox_from_mask ───────────────────────────────────────────────────────────

def test_bbox_correct():
    mask = torch.zeros(10, 10, 10, dtype=torch.long)
    mask[2:5, 3:6, 4:7] = 1
    bbox = bbox_from_mask(mask)
    assert bbox == (2, 3, 4, 5, 6, 7)

def test_bbox_empty_returns_none():
    mask = torch.zeros(10, 10, 10, dtype=torch.long)
    assert bbox_from_mask(mask) is None

def test_bbox_single_voxel():
    mask = torch.zeros(10, 10, 10, dtype=torch.long)
    mask[3, 4, 5] = 1
    bbox = bbox_from_mask(mask)
    assert bbox == (3, 4, 5, 4, 5, 6)  # exclusive z1/y1/x1


# ─── tumor_volume_mm3 ─────────────────────────────────────────────────────────

def test_volume_identity_affine():
    # Identity affine → voxel volume = 1 mm³
    mask = torch.ones(3, 3, 3, dtype=torch.long)
    affine = np.eye(4)
    vol = tumor_volume_mm3(mask, affine)
    assert abs(vol - 27.0) < 1e-4

def test_volume_scaled_affine():
    # 2mm isotropic → voxel volume = 8 mm³
    mask = torch.ones(3, 3, 3, dtype=torch.long)
    affine = np.diag([2.0, 2.0, 2.0, 1.0])
    vol = tumor_volume_mm3(mask, affine)
    assert abs(vol - 216.0) < 1e-4  # 27 voxel * 8 mm³

def test_volume_empty_mask():
    mask = torch.zeros(3, 3, 3, dtype=torch.long)
    affine = np.eye(4)
    vol = tumor_volume_mm3(mask, affine)
    assert vol == 0.0