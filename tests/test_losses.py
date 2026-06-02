import pytest
import torch

from training.losses import (
    CoordinateLoss,
    DiceCELoss,
    DiceLoss,
    FocalLoss,
    LossWeights,
    MultiTaskLoss,
    SegmentationLoss,
    get_lambda_coord,
    one_hot_seg,
)


def make_logits_and_target(
    batch_size: int = 2,
    num_classes: int = 4,
    shape: tuple[int, int, int] = (8, 8, 8),
) -> tuple[torch.Tensor, torch.Tensor]:
    logits = torch.randn(
        batch_size,
        num_classes,
        *shape,
        requires_grad=True,
    )

    target = torch.randint(
        low=0,
        high=num_classes,
        size=(batch_size, *shape),
        dtype=torch.long,
    )

    return logits, target


def test_get_lambda_coord():
    assert get_lambda_coord(epoch=0, warmup_epoch=20, rampup_epoch=40, max_weight=0.1) == 0.0
    assert get_lambda_coord(epoch=20, warmup_epoch=20, rampup_epoch=40, max_weight=0.1) == 0.0
    assert get_lambda_coord(epoch=30, warmup_epoch=20, rampup_epoch=40, max_weight=0.1) == pytest.approx(0.05)
    assert get_lambda_coord(epoch=40, warmup_epoch=20, rampup_epoch=40, max_weight=0.1) == pytest.approx(0.1)


def test_one_hot_seg_shape():
    target = torch.tensor(
        [
            [
                [[0, 1], [2, 3]],
            ]
        ],
        dtype=torch.long,
    )

    out = one_hot_seg(target, num_classes=4)

    assert out.shape == (1, 4, 1, 2, 2)
    assert out.dtype == torch.float32
    assert torch.equal(out.argmax(dim=1), target)


def test_dice_loss_runs_and_backprop():
    logits, target = make_logits_and_target()

    loss_fn = DiceLoss(include_background=False)
    loss = loss_fn(logits, target)

    assert loss.ndim == 0
    assert torch.isfinite(loss)

    loss.backward()
    assert logits.grad is not None
    assert torch.isfinite(logits.grad).all()


def test_dice_ce_loss_runs_and_backprop():
    logits, target = make_logits_and_target()

    loss_fn = DiceCELoss(
        dice_weight=1.0,
        ce_weight=1.0,
        include_background=False,
    )

    loss = loss_fn(logits, target)

    assert loss.ndim == 0
    assert torch.isfinite(loss)

    loss.backward()
    assert logits.grad is not None
    assert torch.isfinite(logits.grad).all()


def test_focal_loss_runs_and_backprop():
    logits, target = make_logits_and_target()

    loss_fn = FocalLoss(gamma=2.0)
    loss = loss_fn(logits, target)

    assert loss.ndim == 0
    assert torch.isfinite(loss)

    loss.backward()
    assert logits.grad is not None
    assert torch.isfinite(logits.grad).all()


def test_segmentation_loss_without_focal():
    logits, target = make_logits_and_target()

    loss_fn = SegmentationLoss(
        dice_weight=1.0,
        ce_weight=1.0,
        focal_weight=0.0,
    )

    out = loss_fn(logits, target)

    assert set(out.keys()) == {"loss_seg", "loss_dice_ce", "loss_focal"}
    assert torch.isfinite(out["loss_seg"])
    assert torch.isfinite(out["loss_dice_ce"])
    assert out["loss_focal"].item() == 0.0


def test_segmentation_loss_with_focal():
    logits, target = make_logits_and_target()

    loss_fn = SegmentationLoss(
        dice_weight=1.0,
        ce_weight=1.0,
        focal_weight=0.5,
    )

    out = loss_fn(logits, target)

    assert torch.isfinite(out["loss_seg"])
    assert torch.isfinite(out["loss_focal"])
    assert out["loss_focal"].item() > 0.0


def test_coordinate_loss_with_et():
    loss_fn = CoordinateLoss(et_weight=0.5)

    pred_wt = torch.tensor([[0.0, 0.0, 0.0], [0.2, 0.2, 0.2]], requires_grad=True)
    pred_et = torch.tensor([[0.0, 0.0, 0.0], [0.2, 0.2, 0.2]], requires_grad=True)

    gt_wt = torch.tensor([[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]])
    gt_et = torch.tensor([[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]])

    has_et = torch.tensor([True, True])

    out = loss_fn(pred_wt, pred_et, gt_wt, gt_et, has_et)

    assert set(out.keys()) == {"loss_coord", "loss_coord_wt", "loss_coord_et"}
    assert torch.isfinite(out["loss_coord"])
    assert out["loss_coord_et"].item() > 0.0

    out["loss_coord"].backward()

    assert pred_wt.grad is not None
    assert pred_et.grad is not None


def test_coordinate_loss_without_et_masks_et_loss():
    loss_fn = CoordinateLoss(et_weight=0.5)

    pred_wt = torch.tensor([[0.1, 0.1, 0.1]], requires_grad=True)
    pred_et = torch.tensor([[1.0, 1.0, 1.0]], requires_grad=True)

    gt_wt = torch.tensor([[0.0, 0.0, 0.0]])
    gt_et = torch.tensor([[0.0, 0.0, 0.0]])

    has_et = torch.tensor([False])

    out = loss_fn(pred_wt, pred_et, gt_wt, gt_et, has_et)

    assert out["loss_coord_et"].item() == 0.0

    out["loss_coord"].backward()

    assert pred_wt.grad is not None
    assert torch.isfinite(pred_wt.grad).all()

    # ET is masked and not connected to loss when no sample has ET.
    assert pred_et.grad is None
def test_multitask_loss_segmentation_only():
    logits, target = make_logits_and_target()

    model_out = {
        "seg_logits": logits,
        "coord_wt": torch.randn(2, 3),
        "coord_et": torch.randn(2, 3),
    }

    batch = {
        "seg": target,
        "wt_centroid_norm": torch.randn(2, 3),
        "et_centroid_norm": torch.randn(2, 3),
        "has_et": torch.tensor([True, False]),
    }

    loss_fn = MultiTaskLoss(
        weights=LossWeights(coord=0.1),
        mode="segmentation_only",
    )

    out = loss_fn(model_out, batch, epoch=0)

    assert torch.isfinite(out["loss_total"])
    assert out["loss_coord"].item() == 0.0
    assert out["lambda_coord"].item() == 0.0


def test_multitask_loss_multitask_before_warmup():
    logits, target = make_logits_and_target()

    model_out = {
        "seg_logits": logits,
        "coord_wt": torch.randn(2, 3, requires_grad=True),
        "coord_et": torch.randn(2, 3, requires_grad=True),
    }

    batch = {
        "seg": target,
        "wt_centroid_norm": torch.randn(2, 3),
        "et_centroid_norm": torch.randn(2, 3),
        "has_et": torch.tensor([True, False]),
    }

    loss_fn = MultiTaskLoss(
        mode="multitask",
        coord_warmup_epoch=20,
        coord_rampup_epoch=40,
        lambda_coord_max=0.1,
    )

    out = loss_fn(model_out, batch, epoch=0)

    assert torch.isfinite(out["loss_total"])
    assert torch.isfinite(out["loss_coord"])
    assert out["lambda_coord"].item() == 0.0


def test_multitask_loss_multitask_after_warmup():
    logits, target = make_logits_and_target()

    model_out = {
        "seg_logits": logits,
        "coord_wt": torch.randn(2, 3, requires_grad=True),
        "coord_et": torch.randn(2, 3, requires_grad=True),
    }

    batch = {
        "seg": target,
        "wt_centroid_norm": torch.randn(2, 3),
        "et_centroid_norm": torch.randn(2, 3),
        "has_et": torch.tensor([True, False]),
    }

    loss_fn = MultiTaskLoss(
        mode="multitask",
        coord_warmup_epoch=20,
        coord_rampup_epoch=40,
        lambda_coord_max=0.1,
    )

    out = loss_fn(model_out, batch, epoch=50)

    assert torch.isfinite(out["loss_total"])
    assert out["lambda_coord"].item() == pytest.approx(0.1)


def test_multitask_loss_invalid_mode_raises():
    with pytest.raises(ValueError):
        MultiTaskLoss(mode="bad_mode")


def test_dice_loss_shape_mismatch_raises():
    logits = torch.randn(2, 4, 8, 8, 8)
    target = torch.randint(0, 4, (2, 7, 8, 8))

    loss_fn = DiceLoss()

    with pytest.raises(ValueError, match="Spatial shape mismatch"):
        loss_fn(logits, target)