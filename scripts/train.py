from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import argparse
import random
import time
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


def auto_discover_cases(root_dir: Path) -> list[str]:
    return sorted(
        d.name for d in root_dir.iterdir()
        if d.is_dir() and not d.name.startswith(".")
    )


def split_case_ids(
    case_ids: list[str],
    val_split: float,
    seed: int,
) -> tuple[list[str], list[str]]:
    rng = np.random.RandomState(seed)
    shuffled = [case_ids[i] for i in rng.permutation(len(case_ids))]
    n_val = max(1, int(len(shuffled) * val_split))
    return shuffled[n_val:], shuffled[:n_val]


def build_datasets(
    cfg: dict[str, Any],
) -> tuple[BraTSDataset, BraTSDataset | None]:
    data_cfg = cfg["data"]
    crop_size = int(data_cfg.get("crop_size", 128))
    normalize = bool(data_cfg.get("normalize", True))
    jitter = int(data_cfg.get("jitter", 0))
    preload = bool(data_cfg.get("preload", False))
    cache_dir = data_cfg.get("cache_dir", None)

    # Pre-split layout: train_dir (+ optional val_dir or val_split)
    if "train_dir" in data_cfg:
        train_root = Path(str(data_cfg["train_dir"]))
        all_ids = auto_discover_cases(train_root)

        val_ds: BraTSDataset | None = None

        if "val_dir" in data_cfg:
            # Separate val directory with labels
            val_root = Path(str(data_cfg["val_dir"]))
            val_ids = auto_discover_cases(val_root)
            train_ids = all_ids
            print(f"Train: {len(train_ids)} cases from {train_root}")
            print(f"Val:   {len(val_ids)} cases from {val_root}")
            val_ds = BraTSDataset(
                root_dir=val_root,
                case_ids=val_ids,
                crop_size=crop_size,
                jitter=0,
                normalize=normalize,
                augment=False,
                preload=preload,
                cache_dir=cache_dir,
            )
        else:
            # Split train_dir by val_split
            val_split = float(data_cfg.get("val_split", 0.0))
            if val_split > 0.0:
                seed = int(cfg["run"].get("seed", 42))
                train_ids, val_ids = split_case_ids(all_ids, val_split, seed)
                print(f"Split: {len(train_ids)} train / {len(val_ids)} val from {train_root}")
                val_ds = BraTSDataset(
                    root_dir=train_root,
                    case_ids=val_ids,
                    crop_size=crop_size,
                    jitter=0,
                    normalize=normalize,
                    augment=False,
                    preload=preload,
                    cache_dir=cache_dir,
                )
            else:
                train_ids = all_ids
                print(f"Train: {len(train_ids)} cases from {train_root} (no val)")

        train_ds = BraTSDataset(
            root_dir=train_root,
            case_ids=train_ids,
            crop_size=crop_size,
            jitter=jitter,
            normalize=normalize,
            augment=True,
            preload=preload,
            cache_dir=cache_dir,
        )
        return train_ds, val_ds

    # Single root_dir with optional val_split (legacy configs)
    root_dir = Path(str(data_cfg["root_dir"]))
    raw_ids = data_cfg["case_ids"]
    if raw_ids == "auto":
        case_ids = auto_discover_cases(root_dir)
        print(f"Auto-discovered {len(case_ids)} cases from {root_dir}")
    else:
        case_ids = list(raw_ids)

    val_split = float(data_cfg.get("val_split", 0.0))
    if val_split > 0.0:
        seed = int(cfg["run"].get("seed", 42))
        train_ids, val_ids = split_case_ids(case_ids, val_split, seed)
        print(f"Split: {len(train_ids)} train / {len(val_ids)} val")
    else:
        train_ids = case_ids
        val_ids = []
        print(f"No val split: {len(train_ids)} train cases")

    train_ds = BraTSDataset(
        root_dir=root_dir,
        case_ids=train_ids,
        crop_size=crop_size,
        jitter=jitter,
        normalize=normalize,
        augment=True,
        preload=preload,
        cache_dir=cache_dir,
    )

    val_ds = None
    if val_ids:
        val_ds = BraTSDataset(
            root_dir=root_dir,
            case_ids=val_ids,
            crop_size=crop_size,
            jitter=0,
            normalize=normalize,
            augment=False,
            preload=preload,
            cache_dir=cache_dir,
        )

    return train_ds, val_ds


def build_model(cfg: dict[str, Any]) -> torch.nn.Module:
    model_cfg = cfg["model"]
    name = model_cfg.get("name", "tiny_unet")
    if name != "tiny_unet":
        raise ValueError(f"Unsupported model name: {name}")
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


def build_scheduler(
    optimizer: torch.optim.Optimizer,
    cfg: dict[str, Any],
    num_epochs: int,
) -> torch.optim.lr_scheduler.LRScheduler | None:
    training_cfg = cfg["training"]
    name = str(training_cfg.get("scheduler", "none"))
    if name == "cosine":
        lr_min = float(training_cfg.get("lr_min", 1e-6))
        return torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=num_epochs,
            eta_min=lr_min,
        )
    return None


def autocast_context(
    device: torch.device,
    precision: str,
) -> AbstractContextManager[Any]:
    if device.type == "cuda" and precision == "fp16":
        return torch.autocast(device_type="cuda", dtype=torch.float16)
    if device.type == "cuda" and precision == "bf16":
        return torch.autocast(device_type="cuda", dtype=torch.bfloat16)
    return nullcontext()


def save_checkpoint(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    metrics: dict[str, float],
    path: Path,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "metrics": metrics,
        },
        path,
    )


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


@torch.no_grad()
def validate(
    model: torch.nn.Module,
    loader: DataLoader,
    criterion: MultiTaskLoss,
    device: torch.device,
    precision: str,
    epoch: int,
) -> dict[str, float]:
    model.eval()

    total_loss = 0.0
    num_batches = 0
    metric_sums: dict[str, float] = {
        "dice_wt": 0.0,
        "dice_tc": 0.0,
        "dice_et": 0.0,
        "coord_wt_mae_norm": 0.0,
        "coord_et_mae_norm": 0.0,
    }

    for batch in loader:
        batch = move_batch_to_device(batch, device)

        with autocast_context(device, precision):
            model_out = model(batch["image"])
            loss_terms = criterion(model_out, batch, epoch=epoch)

        total_loss += float(loss_terms["loss_total"].detach().cpu().item())
        num_batches += 1

        metrics = compute_train_metrics(model_out, batch)
        for key, value in metrics.items():
            metric_sums[key] += value

    n = max(1, num_batches)
    return {
        "val_loss": total_loss / n,
        **{f"val_{k}": v / n for k, v in metric_sums.items()},
    }


def train(cfg: dict[str, Any]) -> None:
    run_cfg = cfg["run"]
    training_cfg = cfg["training"]

    set_seed(int(run_cfg.get("seed", 42)))

    device = resolve_device(str(training_cfg.get("device", "auto")))
    precision = str(training_cfg.get("precision", "fp32"))

    print(f"Run: {run_cfg.get('name', 'unnamed')}")
    print(f"Device: {device} | Precision: {precision}")

    train_ds, val_ds = build_datasets(cfg)

    num_workers = int(training_cfg.get("num_workers", 0))
    batch_size = int(training_cfg.get("batch_size", 1))
    persistent = num_workers > 0

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=device.type == "cuda",
        persistent_workers=persistent,
    )

    val_loader: DataLoader | None = None
    if val_ds is not None:
        val_loader = DataLoader(
            val_ds,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=device.type == "cuda",
            persistent_workers=persistent,
        )

    model = build_model(cfg).to(device)
    criterion = build_loss(cfg)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(training_cfg.get("lr", 1e-4)),
        weight_decay=float(training_cfg.get("weight_decay", 1e-5)),
    )

    max_epochs = int(training_cfg.get("max_epochs", 50))
    scheduler = build_scheduler(optimizer, cfg, max_epochs)
    log_interval = int(training_cfg.get("log_interval", 1))

    ckpt_dir_cfg = training_cfg.get("checkpoint_dir")
    ckpt_dir = Path(str(ckpt_dir_cfg)) if ckpt_dir_cfg else None
    if ckpt_dir:
        print(f"Checkpoints: {ckpt_dir}")

    use_scaler = device.type == "cuda" and precision == "fp16"
    scaler = torch.amp.GradScaler("cuda", enabled=True) if use_scaler else None

    best_score = -1.0
    best_metric_key = "val_dice_wt" if val_loader is not None else "dice_wt"

    for epoch in range(max_epochs):
        model.train()
        t0 = time.perf_counter()

        epoch_loss = 0.0
        num_batches = 0
        term_sums: dict[str, float] = {
            "loss_seg": 0.0,
            "loss_coord": 0.0,
            "loss_coord_wt": 0.0,
            "loss_coord_et": 0.0,
            "lambda_coord": 0.0,
        }
        metric_sums: dict[str, float] = {
            "dice_wt": 0.0,
            "dice_tc": 0.0,
            "dice_et": 0.0,
            "coord_wt_mae_norm": 0.0,
            "coord_et_mae_norm": 0.0,
        }

        for batch in train_loader:
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

        if scheduler is not None:
            scheduler.step()

        n = max(1, num_batches)
        avg_loss = epoch_loss / n
        avg_terms = {k: v / n for k, v in term_sums.items()}
        avg_metrics = {k: v / n for k, v in metric_sums.items()}
        elapsed = time.perf_counter() - t0
        current_lr = optimizer.param_groups[0]["lr"]

        val_metrics: dict[str, float] = {}
        if val_loader is not None:
            val_metrics = validate(model, val_loader, criterion, device, precision, epoch)

        if epoch % log_interval == 0:
            line = (
                f"epoch={epoch:03d} "
                f"loss={avg_loss:.4f} "
                f"seg={avg_terms['loss_seg']:.4f} "
                f"coord={avg_terms['loss_coord']:.4f} "
                f"λ={avg_terms['lambda_coord']:.4f} "
                f"dice_wt={avg_metrics['dice_wt']:.4f} "
                f"dice_tc={avg_metrics['dice_tc']:.4f} "
                f"dice_et={avg_metrics['dice_et']:.4f} "
                f"wt_mae={avg_metrics['coord_wt_mae_norm']:.4f} "
                f"et_mae={avg_metrics['coord_et_mae_norm']:.4f} "
                f"lr={current_lr:.2e} "
                f"t={elapsed:.1f}s"
            )
            if val_metrics:
                line += (
                    f" | val_loss={val_metrics['val_loss']:.4f}"
                    f" val_wt={val_metrics['val_dice_wt']:.4f}"
                    f" val_tc={val_metrics['val_dice_tc']:.4f}"
                    f" val_et={val_metrics['val_dice_et']:.4f}"
                )
            print(line)

        if ckpt_dir is not None:
            all_metrics = {**avg_metrics, **val_metrics}
            monitor = all_metrics.get(best_metric_key, 0.0)

            save_checkpoint(model, optimizer, epoch, all_metrics, ckpt_dir / "last.pt")

            if monitor > best_score:
                best_score = monitor
                save_checkpoint(model, optimizer, epoch, all_metrics, ckpt_dir / "best.pt")
                print(f"  → best saved (epoch={epoch}, {best_metric_key}={best_score:.4f})")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True, help="Path to YAML config file.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    train(cfg)


if __name__ == "__main__":
    main()
