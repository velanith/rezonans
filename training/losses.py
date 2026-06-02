from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass(frozen=True)
class LossWeights:
    dice: float = 1.0
    ce: float = 1.0
    focal: float = 0.0
    coord: float = 0.1
    et_coord: float = 0.5


def get_lambda_coord(
    epoch: int,
    warmup_epoch: int = 20,
    rampup_epoch: int = 40,
    max_weight: float = 0.1,
) -> float:
    if epoch < warmup_epoch:
        return 0.0

    if epoch < rampup_epoch:
        progress = (epoch - warmup_epoch) / max(1, rampup_epoch - warmup_epoch)
        return max_weight * progress

    return max_weight


def one_hot_seg(
    target: torch.Tensor,
    num_classes: int,
) -> torch.Tensor:
    if target.ndim != 4:
        raise ValueError(f"Expected target shape (B,D,H,W), got {tuple(target.shape)}")

    if target.dtype != torch.long:
        target = target.long()

    return F.one_hot(target, num_classes=num_classes).permute(0, 4, 1, 2, 3).float()


class DiceLoss(nn.Module):
    def __init__(
        self,
        include_background: bool = False,
        smooth: float = 1e-5,
    ) -> None:
        super().__init__()
        self.include_background = include_background
        self.smooth = smooth

    def forward(
        self,
        logits: torch.Tensor,
        target: torch.Tensor,
    ) -> torch.Tensor:
        if logits.ndim != 5:
            raise ValueError(f"Expected logits shape (B,C,D,H,W), got {tuple(logits.shape)}")

        if target.ndim != 4:
            raise ValueError(f"Expected target shape (B,D,H,W), got {tuple(target.shape)}")

        if logits.shape[0] != target.shape[0]:
            raise ValueError("Batch size mismatch between logits and target")

        if logits.shape[2:] != target.shape[1:]:
            raise ValueError(
                f"Spatial shape mismatch: logits {tuple(logits.shape[2:])}, "
                f"target {tuple(target.shape[1:])}"
            )

        num_classes = logits.shape[1]

        probs = torch.softmax(logits, dim=1)
        target_oh = one_hot_seg(target, num_classes=num_classes).to(
            dtype=probs.dtype,
            device=probs.device,
        )

        if not self.include_background:
            probs = probs[:, 1:]
            target_oh = target_oh[:, 1:]

        dims = (0, 2, 3, 4)

        intersection = torch.sum(probs * target_oh, dim=dims)
        denominator = torch.sum(probs + target_oh, dim=dims)

        dice = (2.0 * intersection + self.smooth) / (denominator + self.smooth)

        return 1.0 - dice.mean()


class DiceCELoss(nn.Module):
    def __init__(
        self,
        dice_weight: float = 1.0,
        ce_weight: float = 1.0,
        include_background: bool = False,
        class_weights: Optional[torch.Tensor] = None,
    ) -> None:
        super().__init__()

        self.dice_weight = dice_weight
        self.ce_weight = ce_weight
        self.dice = DiceLoss(include_background=include_background)

        if class_weights is not None:
            self.register_buffer("class_weights", class_weights.float())
        else:
            self.class_weights = None

    def forward(
        self,
        logits: torch.Tensor,
        target: torch.Tensor,
    ) -> torch.Tensor:
        loss = logits.new_tensor(0.0)

        if self.dice_weight > 0:
            loss = loss + self.dice_weight * self.dice(logits, target)

        if self.ce_weight > 0:
            weight = self.class_weights
            if weight is not None:
                weight = weight.to(device=logits.device, dtype=logits.dtype)

            ce = F.cross_entropy(
                logits,
                target.long(),
                weight=weight,
            )
            loss = loss + self.ce_weight * ce

        return loss


class FocalLoss(nn.Module):
    def __init__(
        self,
        gamma: float = 2.0,
        alpha: Optional[torch.Tensor] = None,
    ) -> None:
        super().__init__()
        self.gamma = gamma

        if alpha is not None:
            self.register_buffer("alpha", alpha.float())
        else:
            self.alpha = None

    def forward(
        self,
        logits: torch.Tensor,
        target: torch.Tensor,
    ) -> torch.Tensor:
        if logits.ndim != 5:
            raise ValueError(f"Expected logits shape (B,C,D,H,W), got {tuple(logits.shape)}")

        ce = F.cross_entropy(
            logits,
            target.long(),
            reduction="none",
        )

        pt = torch.exp(-ce)
        focal = (1.0 - pt) ** self.gamma * ce

        if self.alpha is not None:
            alpha = self.alpha.to(device=logits.device, dtype=logits.dtype)
            alpha_t = alpha[target.long()]
            focal = alpha_t * focal

        return focal.mean()


class SegmentationLoss(nn.Module):
    def __init__(
        self,
        dice_weight: float = 1.0,
        ce_weight: float = 1.0,
        focal_weight: float = 0.0,
        include_background: bool = False,
        ce_class_weights: Optional[torch.Tensor] = None,
        focal_alpha: Optional[torch.Tensor] = None,
        focal_gamma: float = 2.0,
    ) -> None:
        super().__init__()

        self.focal_weight = focal_weight

        self.dice_ce = DiceCELoss(
            dice_weight=dice_weight,
            ce_weight=ce_weight,
            include_background=include_background,
            class_weights=ce_class_weights,
        )

        self.focal = FocalLoss(
            gamma=focal_gamma,
            alpha=focal_alpha,
        )

    def forward(
        self,
        logits: torch.Tensor,
        target: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        dice_ce = self.dice_ce(logits, target)
        focal = self.focal(logits, target) if self.focal_weight > 0 else logits.new_tensor(0.0)

        total = dice_ce + self.focal_weight * focal

        return {
            "loss_seg": total,
            "loss_dice_ce": dice_ce,
            "loss_focal": focal,
        }


class CoordinateLoss(nn.Module):
    def __init__(
        self,
        et_weight: float = 0.5,
        beta: float = 1.0,
    ) -> None:
        super().__init__()
        self.et_weight = et_weight
        self.beta = beta

    def forward(
        self,
        pred_wt: torch.Tensor,
        pred_et: torch.Tensor,
        gt_wt: torch.Tensor,
        gt_et: torch.Tensor,
        has_et: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        if pred_wt.shape != gt_wt.shape:
            raise ValueError(f"WT shape mismatch: pred {pred_wt.shape}, gt {gt_wt.shape}")

        if pred_et.shape != gt_et.shape:
            raise ValueError(f"ET shape mismatch: pred {pred_et.shape}, gt {gt_et.shape}")

        if pred_wt.ndim != 2 or pred_wt.shape[-1] != 3:
            raise ValueError(f"Expected pred_wt shape (B,3), got {tuple(pred_wt.shape)}")

        if pred_et.ndim != 2 or pred_et.shape[-1] != 3:
            raise ValueError(f"Expected pred_et shape (B,3), got {tuple(pred_et.shape)}")

        has_et = has_et.to(device=pred_wt.device, dtype=torch.bool)

        if has_et.ndim != 1 or has_et.shape[0] != pred_wt.shape[0]:
            raise ValueError(f"Expected has_et shape (B,), got {tuple(has_et.shape)}")

        pred_wt = pred_wt.contiguous()
        pred_et = pred_et.contiguous()
        gt_wt = gt_wt.to(device=pred_wt.device, dtype=pred_wt.dtype).contiguous()
        gt_et = gt_et.to(device=pred_et.device, dtype=pred_et.dtype).contiguous()
        has_et = has_et.contiguous()

        loss_wt = F.smooth_l1_loss(
            pred_wt,
            gt_wt,
            beta=self.beta,
        )

        et_raw = F.smooth_l1_loss(
            pred_et,
            gt_et,
            beta=self.beta,
            reduction="none",
        ).mean(dim=-1)

        has_et_f = has_et.to(dtype=pred_et.dtype)

        if bool(has_et.any()):
            loss_et = (et_raw * has_et_f).sum() / has_et_f.sum()
        else:
            loss_et = pred_et.new_tensor(0.0)

        total = loss_wt + self.et_weight * loss_et

        return {
            "loss_coord": total,
            "loss_coord_wt": loss_wt,
            "loss_coord_et": loss_et,
        }


class MultiTaskLoss(nn.Module):
    def __init__(
        self,
        weights: LossWeights = LossWeights(),
        include_background: bool = False,
        ce_class_weights: Optional[torch.Tensor] = None,
        focal_alpha: Optional[torch.Tensor] = None,
        focal_gamma: float = 2.0,
        coord_warmup_epoch: int = 20,
        coord_rampup_epoch: int = 40,
        lambda_coord_max: float = 0.1,
        mode: str = "multitask",
    ) -> None:
        super().__init__()

        if mode not in {"segmentation_only", "multitask"}:
            raise ValueError(f"Unsupported mode: {mode}")

        self.weights = weights
        self.coord_warmup_epoch = coord_warmup_epoch
        self.coord_rampup_epoch = coord_rampup_epoch
        self.lambda_coord_max = lambda_coord_max
        self.mode = mode

        self.seg_loss = SegmentationLoss(
            dice_weight=weights.dice,
            ce_weight=weights.ce,
            focal_weight=weights.focal,
            include_background=include_background,
            ce_class_weights=ce_class_weights,
            focal_alpha=focal_alpha,
            focal_gamma=focal_gamma,
        )

        self.coord_loss = CoordinateLoss(
            et_weight=weights.et_coord,
        )

    def forward(
        self,
        model_out: Dict[str, torch.Tensor],
        batch: Dict[str, torch.Tensor],
        epoch: int,
    ) -> Dict[str, torch.Tensor]:
        seg_terms = self.seg_loss(
            logits=model_out["seg_logits"],
            target=batch["seg"],
        )

        loss_seg = seg_terms["loss_seg"]

        if self.mode == "segmentation_only":
            zero = loss_seg.new_tensor(0.0)

            return {
                "loss_total": loss_seg,
                **seg_terms,
                "loss_coord": zero,
                "loss_coord_wt": zero,
                "loss_coord_et": zero,
                "lambda_coord": zero,
            }

        coord_terms = self.coord_loss(
            pred_wt=model_out["coord_wt"],
            pred_et=model_out["coord_et"],
            gt_wt=batch["wt_centroid_norm"],
            gt_et=batch["et_centroid_norm"],
            has_et=batch["has_et"],
        )

        lambda_coord = get_lambda_coord(
            epoch=epoch,
            warmup_epoch=self.coord_warmup_epoch,
            rampup_epoch=self.coord_rampup_epoch,
            max_weight=self.lambda_coord_max,
        )

        lambda_coord_t = loss_seg.new_tensor(lambda_coord)

        loss_total = loss_seg + lambda_coord_t * coord_terms["loss_coord"]

        return {
            "loss_total": loss_total,
            **seg_terms,
            **coord_terms,
            "lambda_coord": lambda_coord_t,
        }