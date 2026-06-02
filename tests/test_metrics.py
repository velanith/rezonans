import numpy as np
import pytest
import torch

from training.metrics import (
    coord_mae_mm,
    coord_mae_norm,
    dice_score_binary,
    dice_wt_tc_et,
)


def test_dice_score_binary_perfect_match():
    pred = torch.tensor([1, 0, 1, 0])
    target = torch.tensor([1, 0, 1, 0])

    dice = dice_score_binary(pred, target)

    assert dice.item() == pytest.approx(1.0)


def test_dice_score_binary_no_overlap():
    pred = torch.tensor([1, 1, 0, 0])
    target = torch.tensor([0, 0, 1, 1])

    dice = dice_score_binary(pred, target)

    assert dice.item() == pytest.approx(0.0, abs=1e-8)


def test_dice_score_binary_both_empty():
    pred = torch.zeros((4, 4, 4), dtype=torch.long)
    target = torch.zeros((4, 4, 4), dtype=torch.long)

    dice = dice_score_binary(pred, target)

    assert dice.item() == pytest.approx(1.0)


def test_dice_score_binary_partial_overlap():
    pred = torch.tensor([1, 1, 0, 0])
    target = torch.tensor([1, 0, 1, 0])

    dice = dice_score_binary(pred, target)

    # intersection = 1
    # pred sum = 2, target sum = 2
    # dice = 2 * 1 / 4 = 0.5
    assert dice.item() == pytest.approx(0.5)


def test_dice_wt_tc_et_perfect_match():
    mask = torch.zeros((4, 4, 4), dtype=torch.long)
    mask[1, 1, 1] = 1
    mask[2, 2, 2] = 2
    mask[3, 3, 3] = 3

    out = dice_wt_tc_et(mask, mask)

    assert out["dice_wt"].item() == pytest.approx(1.0)
    assert out["dice_tc"].item() == pytest.approx(1.0)
    assert out["dice_et"].item() == pytest.approx(1.0)


def test_dice_wt_tc_et_region_behavior():
    pred = torch.zeros((4, 4, 4), dtype=torch.long)
    target = torch.zeros((4, 4, 4), dtype=torch.long)

    # Target has NCR, ED, ET.
    target[0, 0, 0] = 1
    target[1, 1, 1] = 2
    target[2, 2, 2] = 3

    # Pred captures NCR and ET but misses ED.
    pred[0, 0, 0] = 1
    pred[2, 2, 2] = 3

    out = dice_wt_tc_et(pred, target)

    # WT: pred=2 voxels, target=3 voxels, intersection=2
    assert out["dice_wt"].item() == pytest.approx(4 / 5)

    # TC: NCR+ET both captured perfectly.
    assert out["dice_tc"].item() == pytest.approx(1.0)

    # ET captured perfectly.
    assert out["dice_et"].item() == pytest.approx(1.0)


def test_dice_wt_tc_et_shape_mismatch_raises():
    pred = torch.zeros((4, 4, 4), dtype=torch.long)
    target = torch.zeros((4, 4, 5), dtype=torch.long)

    with pytest.raises(ValueError, match="Shape mismatch"):
        dice_wt_tc_et(pred, target)


def test_coord_mae_norm_zero():
    pred = torch.tensor([[0.0, 0.0, 0.0]])
    target = torch.tensor([[0.0, 0.0, 0.0]])

    mae = coord_mae_norm(pred, target)

    assert mae.item() == pytest.approx(0.0)


def test_coord_mae_norm_nonzero():
    pred = torch.tensor([[1.0, 0.0, -1.0]])
    target = torch.tensor([[0.0, 0.0, 0.0]])

    mae = coord_mae_norm(pred, target)

    # absolute differences = [1,0,1], mean = 2/3
    assert mae.item() == pytest.approx(2 / 3)


def test_coord_mae_norm_shape_mismatch_raises():
    pred = torch.zeros((1, 3))
    target = torch.zeros((2, 3))

    with pytest.raises(ValueError, match="Shape mismatch"):
        coord_mae_norm(pred, target)


def test_coord_mae_mm_identity_affine_zero_error():
    pred = torch.tensor([[0.0, 0.0, 0.0]])
    target = torch.tensor([[0.0, 0.0, 0.0]])
    origin = torch.tensor([[0, 0, 0]])

    affine = np.eye(4, dtype=np.float32)

    mae = coord_mae_mm(
        pred_norm_zyx=pred,
        target_norm_zyx=target,
        crop_origin_zyx=origin,
        affine=affine,
        crop_shape_zyx=(128, 128, 128),
    )

    assert mae.item() == pytest.approx(0.0)


def test_coord_mae_mm_identity_affine_known_distance():
    # crop shape 3 means normalized -1,0,1 maps to voxel 0,1,2.
    pred = torch.tensor([[1.0, 0.0, 0.0]])    # zyx voxel = [2,1,1]
    target = torch.tensor([[0.0, 0.0, 0.0]])  # zyx voxel = [1,1,1]
    origin = torch.tensor([[0, 0, 0]])

    affine = np.eye(4, dtype=np.float32)

    mae = coord_mae_mm(
        pred_norm_zyx=pred,
        target_norm_zyx=target,
        crop_origin_zyx=origin,
        affine=affine,
        crop_shape_zyx=(3, 3, 3),
    )

    # Difference is 1 voxel in z direction.
    # Identity affine after zyx->xyz gives 1 mm distance.
    assert mae.item() == pytest.approx(1.0)


def test_coord_mae_mm_scaled_affine_known_distance():
    # Difference is 1 voxel in z.
    # zyx -> xyz means this is z world axis.
    # affine z scale = 4 mm.
    pred = torch.tensor([[1.0, 0.0, 0.0]])
    target = torch.tensor([[0.0, 0.0, 0.0]])
    origin = torch.tensor([[0, 0, 0]])

    affine = np.eye(4, dtype=np.float32)
    affine[2, 2] = 4.0

    mae = coord_mae_mm(
        pred_norm_zyx=pred,
        target_norm_zyx=target,
        crop_origin_zyx=origin,
        affine=affine,
        crop_shape_zyx=(3, 3, 3),
    )

    assert mae.item() == pytest.approx(4.0)


def test_coord_mae_mm_with_crop_origin_same_for_pred_and_target():
    pred = torch.tensor([[1.0, 0.0, 0.0]])
    target = torch.tensor([[0.0, 0.0, 0.0]])
    origin = torch.tensor([[10, 20, 30]])

    affine = np.eye(4, dtype=np.float32)

    mae = coord_mae_mm(
        pred_norm_zyx=pred,
        target_norm_zyx=target,
        crop_origin_zyx=origin,
        affine=affine,
        crop_shape_zyx=(3, 3, 3),
    )

    # Origin cancels because both pred and target use same crop origin.
    assert mae.item() == pytest.approx(1.0)


def test_coord_mae_mm_shape_mismatch_raises():
    pred = torch.zeros((1, 3))
    target = torch.zeros((2, 3))
    origin = torch.zeros((1, 3))

    affine = np.eye(4, dtype=np.float32)

    with pytest.raises(ValueError, match="Shape mismatch"):
        coord_mae_mm(pred, target, origin, affine)