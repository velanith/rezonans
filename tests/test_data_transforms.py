import pytest
import torch
from data.transforms import (
    compute_crop_origin,
    crop_or_pad_3d,
    sample_foreground_crop_center,
    foreground_crop,
)


# ─── compute_crop_origin ──────────────────────────────────────────────────────

def test_crop_origin_centered():
    # Merkez crop, sınır yok
    origin = compute_crop_origin((64, 64, 64), crop_size=128, full_shape_zyx=(200, 200, 200))
    assert origin == (0, 0, 0)

def test_crop_origin_no_negative():
    # Center çok küçük → origin 0'dan aşağı düşemez
    origin = compute_crop_origin((5, 5, 5), crop_size=128, full_shape_zyx=(200, 200, 200))
    assert all(o >= 0 for o in origin)

def test_crop_origin_no_overflow():
    # Center çok büyük → origin + crop_size volume'u aşamaz
    origin = compute_crop_origin((190, 190, 190), crop_size=128, full_shape_zyx=(200, 200, 200))
    assert all(o + 128 <= dim for o, dim in zip(origin, (200, 200, 200)))

def test_crop_origin_dim_smaller_than_crop():
    # Volume crop'tan küçük → origin 0
    origin = compute_crop_origin((30, 30, 30), crop_size=128, full_shape_zyx=(64, 64, 64))
    assert origin == (0, 0, 0)

def test_crop_origin_invalid_crop_size():
    with pytest.raises(ValueError):
        compute_crop_origin((10, 10, 10), crop_size=0, full_shape_zyx=(200, 200, 200))


# ─── crop_or_pad_3d ───────────────────────────────────────────────────────────

def test_crop_output_shape_normal():
    vol = torch.zeros(4, 200, 200, 200)
    result = crop_or_pad_3d(vol, origin_zyx=(10, 10, 10), crop_size=128)
    assert result.shape == (4, 128, 128, 128)

def test_crop_output_shape_with_padding():
    # Volume crop'tan küçük → padding gerekir
    vol = torch.zeros(4, 64, 64, 64)
    result = crop_or_pad_3d(vol, origin_zyx=(0, 0, 0), crop_size=128)
    assert result.shape == (4, 128, 128, 128)

def test_crop_padding_value():
    vol = torch.ones(1, 64, 64, 64)
    result = crop_or_pad_3d(vol, origin_zyx=(0, 0, 0), crop_size=128, pad_value=-1.0)
    # Padded bölge -1 olmalı
    assert result[0, 100, 100, 100].item() == -1.0

def test_crop_content_preserved():
    vol = torch.zeros(1, 200, 200, 200)
    vol[0, 50, 60, 70] = 99.0
    result = crop_or_pad_3d(vol, origin_zyx=(10, 10, 10), crop_size=128)
    assert result[0, 40, 50, 60].item() == 99.0  # 50-10, 60-10, 70-10

def test_crop_negative_origin_raises():
    vol = torch.zeros(1, 200, 200, 200)
    with pytest.raises(ValueError):
        crop_or_pad_3d(vol, origin_zyx=(-1, 0, 0), crop_size=128)

def test_crop_3d_mask_no_channel():
    # Mask (D,H,W) — channel boyutu yok
    vol = torch.zeros(200, 200, 200)
    result = crop_or_pad_3d(vol, origin_zyx=(0, 0, 0), crop_size=128)
    assert result.shape == (128, 128, 128)


# ─── sample_foreground_crop_center ───────────────────────────────────────────

def test_sample_returns_foreground_voxel():
    mask = torch.zeros(50, 50, 50, dtype=torch.long)
    mask[10, 20, 30] = 1
    # Jitter 0 → tam o voxel dönmeli
    center = sample_foreground_crop_center(mask, jitter=0)
    assert center == (10, 20, 30)

def test_sample_within_bounds_with_jitter():
    mask = torch.zeros(50, 50, 50, dtype=torch.long)
    mask[25, 25, 25] = 1
    for _ in range(20):
        center = sample_foreground_crop_center(mask, jitter=32)
        d, h, w = 50, 50, 50
        assert 0 <= center[0] < d
        assert 0 <= center[1] < h
        assert 0 <= center[2] < w

def test_sample_empty_mask_still_returns_valid():
    # Foreground yok → random center, ama bounds içinde
    mask = torch.zeros(50, 50, 50, dtype=torch.long)
    center = sample_foreground_crop_center(mask, jitter=0)
    assert all(0 <= c < 50 for c in center)

def test_sample_negative_jitter_raises():
    mask = torch.zeros(10, 10, 10, dtype=torch.long)
    with pytest.raises(ValueError):
        sample_foreground_crop_center(mask, jitter=-1)


# ─── foreground_crop ─────────────────────────────────────────────────────────

def test_foreground_crop_output_shape():
    image = torch.zeros(4, 155, 240, 240)
    mask = torch.zeros(155, 240, 240, dtype=torch.long)
    mask[80, 120, 120] = 1
    result = foreground_crop(image, mask, crop_size=128, jitter=0)
    assert result["image"].shape == (4, 128, 128, 128)
    assert result["mask"].shape == (128, 128, 128)

def test_foreground_crop_origin_type():
    image = torch.zeros(4, 155, 240, 240)
    mask = torch.zeros(155, 240, 240, dtype=torch.long)
    mask[80, 120, 120] = 1
    result = foreground_crop(image, mask, crop_size=128, jitter=0)
    oz, oy, ox = result["crop_origin_zyx"]
    assert isinstance(oz, int)
    assert isinstance(oy, int)
    assert isinstance(ox, int)

def test_foreground_crop_origin_valid():
    image = torch.zeros(4, 155, 240, 240)
    mask = torch.zeros(155, 240, 240, dtype=torch.long)
    mask[80, 120, 120] = 1
    result = foreground_crop(image, mask, crop_size=128, jitter=0)
    oz, oy, ox = result["crop_origin_zyx"]
    assert oz >= 0 and oy >= 0 and ox >= 0
    assert oz + 128 <= 155
    assert oy + 128 <= 240
    assert ox + 128 <= 240

def test_foreground_crop_mask_dtype():
    image = torch.zeros(4, 155, 240, 240)
    mask = torch.zeros(155, 240, 240, dtype=torch.long)
    mask[80, 120, 120] = 2
    result = foreground_crop(image, mask, crop_size=128, jitter=0)
    assert result["mask"].dtype == torch.long

def test_foreground_crop_wrong_image_dims():
    image = torch.zeros(155, 240, 240)  # C boyutu yok
    mask = torch.zeros(155, 240, 240, dtype=torch.long)
    with pytest.raises(ValueError):
        foreground_crop(image, mask)

def test_foreground_crop_shape_mismatch():
    image = torch.zeros(4, 155, 240, 240)
    mask = torch.zeros(128, 128, 128, dtype=torch.long)  # farklı shape
    with pytest.raises(ValueError):
        foreground_crop(image, mask)


def test_compute_crop_origin_centered():
    origin = compute_crop_origin(
        center_zyx=(64, 64, 64),
        crop_size=32,
        full_shape_zyx=(128, 128, 128),
    )

    assert origin == (48, 48, 48)


def test_compute_crop_origin_near_start_boundary():
    origin = compute_crop_origin(
        center_zyx=(5, 6, 7),
        crop_size=32,
        full_shape_zyx=(128, 128, 128),
    )

    assert origin == (0, 0, 0)


def test_compute_crop_origin_near_end_boundary():
    origin = compute_crop_origin(
        center_zyx=(120, 121, 122),
        crop_size=32,
        full_shape_zyx=(128, 128, 128),
    )

    assert origin == (96, 96, 96)


def test_compute_crop_origin_dim_smaller_than_crop():
    origin = compute_crop_origin(
        center_zyx=(10, 20, 30),
        crop_size=64,
        full_shape_zyx=(32, 128, 40),
    )

    assert origin == (0, 0, 0)


def test_crop_or_pad_3d_image_no_padding():
    volume = torch.arange(1 * 10 * 20 * 30).reshape(1, 10, 20, 30)

    crop = crop_or_pad_3d(
        volume=volume,
        origin_zyx=(2, 4, 6),
        crop_size=4,
    )

    expected = volume[:, 2:6, 4:8, 6:10]

    assert crop.shape == (1, 4, 4, 4)
    assert torch.equal(crop, expected)


def test_crop_or_pad_3d_mask_no_padding():
    volume = torch.arange(10 * 20 * 30).reshape(10, 20, 30)

    crop = crop_or_pad_3d(
        volume=volume,
        origin_zyx=(2, 4, 6),
        crop_size=4,
    )

    expected = volume[2:6, 4:8, 6:10]

    assert crop.shape == (4, 4, 4)
    assert torch.equal(crop, expected)


def test_crop_or_pad_3d_with_padding():
    volume = torch.ones((1, 5, 6, 7))

    crop = crop_or_pad_3d(
        volume=volume,
        origin_zyx=(3, 4, 5),
        crop_size=4,
        pad_value=0.0,
    )

    assert crop.shape == (1, 4, 4, 4)

    # Real available area:
    # z: 3:5 -> 2 voxels
    # y: 4:6 -> 2 voxels
    # x: 5:7 -> 2 voxels
    assert torch.equal(crop[:, :2, :2, :2], torch.ones((1, 2, 2, 2)))
    assert crop[:, 2:, :, :].sum() == 0
    assert crop[:, :, 2:, :].sum() == 0
    assert crop[:, :, :, 2:].sum() == 0


def test_sample_foreground_crop_center_without_jitter():
    torch.manual_seed(0)

    mask = torch.zeros((16, 16, 16), dtype=torch.long)
    mask[5, 6, 7] = 1

    center = sample_foreground_crop_center(mask, jitter=0)

    assert center == (5, 6, 7)


def test_sample_foreground_crop_center_with_jitter_is_in_bounds():
    torch.manual_seed(0)

    mask = torch.zeros((16, 16, 16), dtype=torch.long)
    mask[0, 0, 0] = 1

    for _ in range(20):
        z, y, x = sample_foreground_crop_center(mask, jitter=32)

        assert 0 <= z < 16
        assert 0 <= y < 16
        assert 0 <= x < 16


def test_sample_foreground_crop_center_empty_mask_is_in_bounds():
    torch.manual_seed(0)

    mask = torch.zeros((8, 9, 10), dtype=torch.long)

    for _ in range(20):
        z, y, x = sample_foreground_crop_center(mask)

        assert 0 <= z < 8
        assert 0 <= y < 9
        assert 0 <= x < 10


def test_foreground_crop_shapes_and_origin():
    torch.manual_seed(0)

    image = torch.zeros((4, 32, 32, 32))
    mask = torch.zeros((32, 32, 32), dtype=torch.long)

    image[:, 10, 11, 12] = 5.0
    mask[10, 11, 12] = 1

    out = foreground_crop(
        image=image,
        mask=mask,
        crop_size=16,
        jitter=0,
    )

    image_crop = out["image"]
    mask_crop = out["mask"]
    origin = out["crop_origin_zyx"]

    assert isinstance(image_crop, torch.Tensor)
    assert isinstance(mask_crop, torch.Tensor)
    assert origin == (2, 3, 4)

    assert image_crop.shape == (4, 16, 16, 16)
    assert mask_crop.shape == (16, 16, 16)

    # Original foreground voxel (10,11,12)
    # Crop origin (2,3,4)
    # Crop-local position should be (8,8,8)
    assert torch.equal(mask_crop[8, 8, 8], torch.tensor(1))
    assert torch.equal(image_crop[:, 8, 8, 8], torch.full((4,), 5.0))


def test_foreground_crop_with_padding_when_volume_smaller_than_crop():
    torch.manual_seed(0)

    image = torch.ones((4, 8, 8, 8))
    mask = torch.zeros((8, 8, 8), dtype=torch.long)
    mask[4, 4, 4] = 1

    out = foreground_crop(
        image=image,
        mask=mask,
        crop_size=16,
        jitter=0,
    )

    image_crop = out["image"]
    mask_crop = out["mask"]
    origin = out["crop_origin_zyx"]

    assert isinstance(image_crop, torch.Tensor)
    assert isinstance(mask_crop, torch.Tensor)

    assert origin == (0, 0, 0)
    assert image_crop.shape == (4, 16, 16, 16)
    assert mask_crop.shape == (16, 16, 16)

    assert torch.equal(image_crop[:, :8, :8, :8], torch.ones((4, 8, 8, 8)))
    assert torch.equal(mask_crop[:8, :8, :8], mask)
    assert image_crop[:, 8:, :, :].sum() == 0
    assert image_crop[:, :, 8:, :].sum() == 0
    assert image_crop[:, :, :, 8:].sum() == 0


def test_foreground_crop_raises_on_shape_mismatch():
    image = torch.zeros((4, 16, 16, 16))
    mask = torch.zeros((15, 16, 16), dtype=torch.long)

    try:
        foreground_crop(image=image, mask=mask, crop_size=8)
    except ValueError as exc:
        assert "does not match mask shape" in str(exc)
    else:
        raise AssertionError("Expected ValueError")
    