from __future__ import annotations

import argparse
import random
from contextlib import AbstractContextManager, nullcontext
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader

from data.dataset import BraTSDataset
from models.tiny_unet import TinyUNet3D
from training.losses import LossWeights, MultiTaskLoss
from training.metrics import coord_mae_norm, dice_wt_tc_et


def load_config(path: str | Path) -> dict[str, Any]:
    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(f"Config not found: {path}")

    with path.open("r") as f:
        cfg = yaml.safe_load(f)

    if not isinstance(cfg, dict):
        raise ValueError(f"Config must be a dictionary, got {type(cfg)}")

    return cfg


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_device(device_name: str) -> torch.device:
    if device_name == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")

    return torch.device(device_name)


def move_batch_to_device(
    batch: dict[str, Any],
    device: torch.device,
) -> dict[str, Any]:
    out: dict[str, Any] = {}

    for key, value in batch.items():
        if isinstance(value, torch.Tensor):
            out[key] = value.to(device)
        else:
            out[key] = value

    return out


def build_dataset(cfg: dict[str, Any]) -> BraTSDataset:
    data_cfg = cfg["data"]

    return BraTSDataset(
        root_dir=data_cfg["root_dir"],
        case_ids=list(data_cfg["case_ids"]),
        crop_size=int(data_cfg.get("crop_size", 128)),
        jitter=int(data_cfg.get("jitter", 0)),
        normalize=bool(data_cfg.get("normalize", True)),
    )


def build_model(cfg: dict[str, Any]) -> torch.nn.Module:
    model_cfg = cfg["model"]

    name = model_cfg.get("name", "tiny_unet")

    if name != "tiny_unet":
        raise ValueError(f"Unsupported model name for now: {name}")

    return TinyUNet3D(
        in_channels=int(model_cfg.get("in_channels", 4)),
        out_channels=int(model_cfg.get("out_channels", 4)),
        base_channels=int(model_cfg.get("base_channels", 4)),
        use_coord_head=bool(model_cfg.get("use_coord_head", True)),
    )


def build_loss(cfg: dict[str, Any]) -> MultiTaskLoss:
    loss_cfg = cfg["loss"]
    training_cfg = cfg["training"]

    weights = LossWeights(
        dice=float(loss_cfg.get("dice", 1.0)),
        ce=float(loss_cfg.get("ce", 1.0)),
        focal=float(loss_cfg.get("focal", 0.0)),
        coord=float(loss_cfg.get("coord", 0.1)),
        et_coord=float(loss_cfg.get("et_coord", 0.5)),
    )

    return MultiTaskLoss(
        weights=weights,
        mode=str(training_cfg.get("mode", "multitask")),
        coord_warmup_epoch=int(loss_cfg.get("coord_warmup_epoch", 20)),
        coord_rampup_epoch=int(loss_cfg.get("coord_rampup_epoch", 40)),
        lambda_coord_max=float(loss_cfg.get("lambda_coord_max", 0.1)),
    )


def autocast_context(
    device: torch.device,
    precision: str,
) -> AbstractContextManager[Any]:
    if device.type == "cuda" and precision == "fp16":
        return torch.autocast(device_type="cuda", dtype=torch.float16)

    if device.type == "cuda" and precision == "bf16":
        return torch.autocast(device_type="cuda", dtype=torch.bfloat16)

    return nullcontext()


@torch.no_grad()
def compute_train_metrics(
    model_out: dict[str, torch.Tensor],
    batch: dict[str, torch.Tensor],
) -> dict[str, float]:
    seg_logits = model_out["seg_logits"]
    pred_mask = torch.argmax(seg_logits, dim=1)

    batch_size = pred_mask.shape[0]

    dice_wt_sum = 0.0
    dice_tc_sum = 0.0
    dice_et_sum = 0.0

    for i in range(batch_size):
        dice = dice_wt_tc_et(
            pred_mask=pred_mask[i].detach().cpu(),
            target_mask=batch["seg"][i].detach().cpu(),
        )

        dice_wt_sum += float(dice["dice_wt"].item())
        dice_tc_sum += float(dice["dice_tc"].item())
        dice_et_sum += float(dice["dice_et"].item())

    wt_mae = coord_mae_norm(
        model_out["coord_wt"].detach().cpu(),
        batch["wt_centroid_norm"].detach().cpu(),
    )

    et_mae = coord_mae_norm(
        model_out["coord_et"].detach().cpu(),
        batch["et_centroid_norm"].detach().cpu(),
    )

    return {
        "dice_wt": dice_wt_sum / batch_size,
        "dice_tc": dice_tc_sum / batch_size,
        "dice_et": dice_et_sum / batch_size,
        "coord_wt_mae_norm": float(wt_mae.item()),
        "coord_et_mae_norm": float(et_mae.item()),
    }


def train(cfg: dict[str, Any]) -> None:
    run_cfg = cfg["run"]
    training_cfg = cfg["training"]

    set_seed(int(run_cfg.get("seed", 42)))

    device = resolve_device(str(training_cfg.get("device", "auto")))
    precision = str(training_cfg.get("precision", "fp32"))

    print(f"Run: {run_cfg.get('name', 'unnamed')}")
    print(f"Device: {device}")
    print(f"Precision: {precision}")

    dataset = build_dataset(cfg)

    loader = DataLoader(
        dataset,
        batch_size=int(training_cfg.get("batch_size", 1)),
        shuffle=True,
        num_workers=int(training_cfg.get("num_workers", 0)),
        pin_memory=device.type == "cuda",
    )

    model = build_model(cfg).to(device)
    criterion = build_loss(cfg)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(training_cfg.get("lr", 1e-4)),
        weight_decay=float(training_cfg.get("weight_decay", 1e-5)),
    )

    max_epochs = int(training_cfg.get("max_epochs", 50))
    log_interval = int(training_cfg.get("log_interval", 1))

    use_scaler = device.type == "cuda" and precision == "fp16"
    scaler = torch.amp.GradScaler("cuda", enabled=True) if use_scaler else None

    for epoch in range(max_epochs):
        model.train()

        epoch_loss = 0.0
        num_batches = 0

        term_sums = {
            "loss_seg": 0.0,
            "loss_coord": 0.0,
            "loss_coord_wt": 0.0,
            "loss_coord_et": 0.0,
            "lambda_coord": 0.0,
        }

        metric_sums = {
            "dice_wt": 0.0,
            "dice_tc": 0.0,
            "dice_et": 0.0,
            "coord_wt_mae_norm": 0.0,
            "coord_et_mae_norm": 0.0,
        }

        for batch in loader:
            batch = move_batch_to_device(batch, device)

            optimizer.zero_grad(set_to_none=True)

            with autocast_context(device, precision):
                model_out = model(batch["image"])
                loss_terms = criterion(model_out, batch, epoch=epoch)
                loss: torch.Tensor = loss_terms["loss_total"]

            if use_scaler:
                if scaler is None:
                    raise RuntimeError("GradScaler is None while use_scaler=True")

                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                optimizer.step()

            epoch_loss += float(loss.detach().cpu().item())
            num_batches += 1

            for key in term_sums:
                term_sums[key] += float(loss_terms[key].detach().cpu().item())

            metrics = compute_train_metrics(model_out, batch)

            for key, value in metrics.items():
                metric_sums[key] += value

        avg_loss = epoch_loss / max(1, num_batches)

        avg_terms = {
            key: value / max(1, num_batches)
            for key, value in term_sums.items()
        }

        avg_metrics = {
            key: value / max(1, num_batches)
            for key, value in metric_sums.items()
        }

        if epoch % log_interval == 0:
            print(
                f"epoch={epoch:03d} "
                f"loss={avg_loss:.4f} "
                f"seg={avg_terms['loss_seg']:.4f} "
                f"coord={avg_terms['loss_coord']:.4f} "
                f"lambda_coord={avg_terms['lambda_coord']:.4f} "
                f"dice_wt={avg_metrics['dice_wt']:.4f} "
                f"dice_tc={avg_metrics['dice_tc']:.4f} "
                f"dice_et={avg_metrics['dice_et']:.4f} "
                f"wt_mae={avg_metrics['coord_wt_mae_norm']:.4f} "
                f"et_mae={avg_metrics['coord_et_mae_norm']:.4f}"
            )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to YAML config file.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    train(cfg)


if __name__ == "__main__":
    main()