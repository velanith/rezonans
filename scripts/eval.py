from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import argparse
import csv

import torch

from data.dataset import load_brats_case, zscore_normalize
from data.utils import to_regions
from training.builder import auto_discover_cases, build_model
from training.metrics import coord_mae_mm, dice_score_binary
from training.predictor import predict_case
from training.utils import load_config, resolve_device


def eval_case(
    pred_seg: torch.Tensor,       # (D, H, W) full volume, labels 0-3
    gt_seg: torch.Tensor,         # (D, H, W) full volume, labels 0-3
    pred_wt_mm: list[float],
    pred_et_mm: list[float],
    gt_wt_mm: list[float],
    gt_et_mm: list[float],
) -> dict[str, float]:
    pred_r = to_regions(pred_seg)
    gt_r = to_regions(gt_seg)

    dice_wt = float(dice_score_binary(pred_r["wt"], gt_r["wt"]).item())
    dice_tc = float(dice_score_binary(pred_r["tc"], gt_r["tc"]).item())
    dice_et = float(dice_score_binary(pred_r["et"], gt_r["et"]).item())

    wt_err = float(torch.linalg.norm(
        torch.tensor(pred_wt_mm) - torch.tensor(gt_wt_mm)
    ).item())
    et_err = float(torch.linalg.norm(
        torch.tensor(pred_et_mm) - torch.tensor(gt_et_mm)
    ).item())

    return {
        "dice_wt": dice_wt,
        "dice_tc": dice_tc,
        "dice_et": dice_et,
        "wt_err_mm": wt_err,
        "et_err_mm": et_err,
    }


def load_gt_from_cache(cache_dir: Path, case_id: str) -> dict:
    """Load GT mask and affine from precomputed .pt cache file."""
    cache_path = cache_dir / f"{case_id}.pt"
    saved = torch.load(cache_path, map_location="cpu", weights_only=True)
    return {
        "mask": saved["mask"].long(),
        "affine": saved["affine"].numpy(),
    }


def run_eval(args: argparse.Namespace) -> None:
    cfg = load_config(args.config)
    device = resolve_device(args.device)
    precision = args.precision
    crop_size = int(cfg["data"].get("crop_size", 128))
    data_dir = Path(args.data_dir)
    cache_dir = Path(args.cache_dir) if args.cache_dir else None

    model = load_checkpoint(Path(args.checkpoint), cfg, device)

    if args.case_ids:
        allowed = set(Path(args.case_ids).read_text().splitlines())
        case_dirs = sorted(d for d in data_dir.iterdir() if d.is_dir() and d.name in allowed)
    else:
        case_dirs = sorted(d for d in data_dir.iterdir() if d.is_dir())
    print(f"Evaluating {len(case_dirs)} cases...")

    rows: list[dict] = []

    for case_dir in case_dirs:
        case_id = case_dir.name

        try:
            pred = predict_case(model, case_dir, device, precision, crop_size)
        except Exception as e:
            print(f"  SKIP {case_id}: {e}")
            continue

        # GT: prefer cache, fall back to raw NIfTI
        from data.utils import compute_centroid_with_fallback, apply_affine_zyx

        if cache_dir and (cache_dir / f"{case_id}.pt").exists():
            gt = load_gt_from_cache(cache_dir, case_id)
            gt_seg = gt["mask"]
            affine = gt["affine"]
        else:
            gt_case = load_brats_case(case_dir)
            gt_seg = gt_case["mask"]
            affine = gt_case["affine"]

        gt_centroids = compute_centroid_with_fallback(gt_seg)
        gt_wt_vox = gt_centroids["wt_centroid_voxel"]
        gt_et_vox = gt_centroids["et_centroid_voxel"]

        gt_wt_mm = apply_affine_zyx(affine, gt_wt_vox.unsqueeze(0)).squeeze(0).tolist()
        gt_et_mm = apply_affine_zyx(affine, gt_et_vox.unsqueeze(0)).squeeze(0).tolist()

        metrics = eval_case(
            pred_seg=pred.seg_full,
            gt_seg=gt_seg,
            pred_wt_mm=pred.wt_centroid_mm,
            pred_et_mm=pred.et_centroid_mm,
            gt_wt_mm=[round(float(v), 3) for v in gt_wt_mm],
            gt_et_mm=[round(float(v), 3) for v in gt_et_mm],
        )

        row = {"case_id": case_id, **metrics}
        rows.append(row)

        print(
            f"  {case_id} | "
            f"Dice WT={metrics['dice_wt']:.4f} "
            f"TC={metrics['dice_tc']:.4f} "
            f"ET={metrics['dice_et']:.4f} | "
            f"WT err={metrics['wt_err_mm']:.1f}mm "
            f"ET err={metrics['et_err_mm']:.1f}mm"
        )

    if not rows:
        print("No cases evaluated.")
        return

    keys = ["dice_wt", "dice_tc", "dice_et", "wt_err_mm", "et_err_mm"]
    print("\n--- Summary ---")
    for k in keys:
        vals = [r[k] for r in rows]
        mean = sum(vals) / len(vals)
        std = (sum((v - mean) ** 2 for v in vals) / len(vals)) ** 0.5
        print(f"  {k:20s}: {mean:.4f} ± {std:.4f}")

    if args.output:
        out_path = Path(args.output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with out_path.open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["case_id"] + keys)
            writer.writeheader()
            writer.writerows(rows)
        print(f"\nSaved to {out_path}")


def load_checkpoint(checkpoint_path: Path, cfg: dict, device: torch.device) -> torch.nn.Module:
    model = build_model(cfg).to(device)
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=True)
    model.load_state_dict(ckpt["model_state_dict"])
    print(f"Loaded checkpoint from epoch {ckpt.get('epoch', '?')}")
    return model


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a trained checkpoint on a labeled dataset.")
    parser.add_argument("--checkpoint", required=True, help="Path to best.pt")
    parser.add_argument("--config", required=True, help="Path to the config used for training")
    parser.add_argument("--data_dir", required=True, help="Directory of labeled cases")
    parser.add_argument("--cache_dir", default=None, help="Precomputed .pt cache directory")
    parser.add_argument("--case_ids", default=None, help="Text file with one case ID per line to filter")
    parser.add_argument("--output", default=None, help="Optional CSV output path")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--precision", default="fp16", choices=["fp32", "fp16", "bf16"])
    return parser.parse_args()


def main() -> None:
    run_eval(parse_args())


if __name__ == "__main__":
    main()
