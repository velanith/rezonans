from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import argparse
import json

import nibabel as nib
import numpy as np
import torch

from training.builder import build_model
from training.predictor import predict_case
from training.utils import load_config, resolve_device


def save_seg_nifti(seg: torch.Tensor, affine: np.ndarray, path: Path) -> None:
    """Save (D,H,W) segmentation mask as NIfTI (H,W,D) with original affine."""
    path.parent.mkdir(parents=True, exist_ok=True)
    data_hwd = np.transpose(seg.numpy().astype(np.int16), (1, 2, 0))
    nib.save(nib.Nifti1Image(data_hwd, affine), str(path))


def load_checkpoint(checkpoint_path: Path, cfg: dict, device: torch.device) -> torch.nn.Module:
    model = build_model(cfg).to(device)
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=True)
    model.load_state_dict(ckpt["model_state_dict"])
    print(f"Loaded checkpoint from epoch {ckpt.get('epoch', '?')}")
    return model


def run_predict(args: argparse.Namespace) -> None:
    cfg = load_config(args.config)
    device = resolve_device(args.device)
    precision = args.precision
    crop_size = int(cfg["data"].get("crop_size", 128))
    output_dir = Path(args.output)

    model = load_checkpoint(Path(args.checkpoint), cfg, device)

    input_path = Path(args.input)

    # Single case dir or directory of cases
    if (input_path / f"{input_path.name}_t1.nii.gz").exists():
        case_dirs = [input_path]
    else:
        case_dirs = sorted(d for d in input_path.iterdir() if d.is_dir())

    print(f"Running inference on {len(case_dirs)} case(s)...")

    for case_dir in case_dirs:
        pred = predict_case(model, case_dir, device, precision, crop_size)

        case_out = output_dir / pred.case_id
        case_out.mkdir(parents=True, exist_ok=True)

        # Load affine for NIfTI saving
        from data.dataset import load_brats_case
        affine = load_brats_case(case_dir)["affine"]

        seg_path = case_out / f"{pred.case_id}_pred_seg.nii.gz"
        save_seg_nifti(pred.seg_full, affine, seg_path)

        result = {
            "case_id": pred.case_id,
            "wt_centroid_mm": pred.wt_centroid_mm,
            "et_centroid_mm": pred.et_centroid_mm,
            "wt_centroid_voxel_zyx": pred.wt_centroid_voxel_zyx,
            "et_centroid_voxel_zyx": pred.et_centroid_voxel_zyx,
            "confidence": {
                "wt_voxel_count": pred.wt_voxel_count,
                "mean_wt_confidence": pred.mean_wt_confidence,
                "review_recommended": pred.review_recommended,
            },
        }

        json_path = case_out / f"{pred.case_id}_pred.json"
        json_path.write_text(json.dumps(result, indent=2))

        review_flag = " ⚠ REVIEW" if pred.review_recommended else ""
        print(
            f"  {pred.case_id} → "
            f"WT mm: {pred.wt_centroid_mm} | "
            f"ET mm: {pred.et_centroid_mm} | "
            f"conf={pred.mean_wt_confidence:.2f} vox={pred.wt_voxel_count}"
            f"{review_flag}"
        )

    print(f"Done. Results saved to {output_dir}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run inference with a trained checkpoint.")
    parser.add_argument("--checkpoint", required=True, help="Path to best.pt")
    parser.add_argument("--config", required=True, help="Path to the config used for training")
    parser.add_argument("--input", required=True, help="Case directory or directory of cases")
    parser.add_argument("--output", required=True, help="Output directory")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--precision", default="fp16", choices=["fp32", "fp16", "bf16"])
    return parser.parse_args()


def main() -> None:
    run_predict(parse_args())


if __name__ == "__main__":
    main()
