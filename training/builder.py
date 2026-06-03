from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader

from data.dataset import BraTSDataset
from models.tiny_unet import TinyUNet3D
from models.ttfields_unetr import TTFieldsUNETR
from training.losses import LossWeights, MultiTaskLoss


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


def _make_dataset(
    root_dir: Path,
    case_ids: list[str],
    crop_size: int,
    jitter: int,
    normalize: bool,
    augment: bool,
    preload: bool,
    cache_dir: str | None,
) -> BraTSDataset:
    return BraTSDataset(
        root_dir=root_dir,
        case_ids=case_ids,
        crop_size=crop_size,
        jitter=jitter,
        normalize=normalize,
        augment=augment,
        preload=preload,
        cache_dir=cache_dir,
    )


def build_datasets(
    cfg: dict[str, Any],
) -> tuple[BraTSDataset, BraTSDataset | None]:
    data_cfg = cfg["data"]
    crop_size = int(data_cfg.get("crop_size", 128))
    normalize = bool(data_cfg.get("normalize", True))
    jitter = int(data_cfg.get("jitter", 0))
    preload = bool(data_cfg.get("preload", False))
    cache_dir = data_cfg.get("cache_dir", None)

    def make(root: Path, ids: list[str], augment: bool, jitter_: int) -> BraTSDataset:
        return _make_dataset(root, ids, crop_size, jitter_, normalize, augment, preload, cache_dir)

    if "train_dir" in data_cfg:
        train_root = Path(str(data_cfg["train_dir"]))
        all_ids = auto_discover_cases(train_root)
        val_ds: BraTSDataset | None = None

        if "val_dir" in data_cfg:
            val_root = Path(str(data_cfg["val_dir"]))
            val_ids = auto_discover_cases(val_root)
            train_ids = all_ids
            print(f"Train: {len(train_ids)} cases | Val: {len(val_ids)} cases")
            val_ds = make(val_root, val_ids, augment=False, jitter_=0)
        else:
            val_split = float(data_cfg.get("val_split", 0.0))
            if val_split > 0.0:
                seed = int(cfg["run"].get("seed", 42))
                train_ids, val_ids = split_case_ids(all_ids, val_split, seed)
                print(f"Split: {len(train_ids)} train / {len(val_ids)} val")
                val_ds = make(train_root, val_ids, augment=False, jitter_=0)
            else:
                train_ids = all_ids
                print(f"Train: {len(train_ids)} cases (no val)")

        return make(train_root, train_ids, augment=True, jitter_=jitter), val_ds

    # Legacy: single root_dir
    root_dir = Path(str(data_cfg["root_dir"]))
    raw_ids = data_cfg["case_ids"]
    if raw_ids == "auto":
        case_ids = auto_discover_cases(root_dir)
        print(f"Auto-discovered {len(case_ids)} cases")
    else:
        case_ids = list(raw_ids)

    val_split = float(data_cfg.get("val_split", 0.0))
    if val_split > 0.0:
        seed = int(cfg["run"].get("seed", 42))
        train_ids, val_ids = split_case_ids(case_ids, val_split, seed)
        print(f"Split: {len(train_ids)} train / {len(val_ids)} val")
        return (
            make(root_dir, train_ids, augment=True, jitter_=jitter),
            make(root_dir, val_ids, augment=False, jitter_=0),
        )

    print(f"Train: {len(case_ids)} cases (no val)")
    return make(root_dir, case_ids, augment=True, jitter_=jitter), None


def build_loaders(
    train_ds: BraTSDataset,
    val_ds: BraTSDataset | None,
    cfg: dict[str, Any],
    device: torch.device,
) -> tuple[DataLoader, DataLoader | None]:
    training_cfg = cfg["training"]
    batch_size = int(training_cfg.get("batch_size", 1))
    num_workers = int(training_cfg.get("num_workers", 0))
    persistent = num_workers > 0
    pin = device.type == "cuda"

    def worker_init(worker_id: int) -> None:
        import torch
        torch.set_num_threads(1)

    init_fn = worker_init if num_workers > 0 else None

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin,
        persistent_workers=persistent,
        prefetch_factor=4 if num_workers > 0 else None,
        worker_init_fn=init_fn,
    )

    val_loader: DataLoader | None = None
    if val_ds is not None:
        val_loader = DataLoader(
            val_ds,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=pin,
            persistent_workers=persistent,
            prefetch_factor=4 if num_workers > 0 else None,
            worker_init_fn=init_fn,
        )

    return train_loader, val_loader


def build_model(cfg: dict[str, Any]) -> torch.nn.Module:
    model_cfg = cfg["model"]
    name = model_cfg.get("name", "tiny_unet")

    if name == "tiny_unet":
        return TinyUNet3D(
            in_channels=int(model_cfg.get("in_channels", 4)),
            out_channels=int(model_cfg.get("out_channels", 4)),
            base_channels=int(model_cfg.get("base_channels", 4)),
            use_coord_head=bool(model_cfg.get("use_coord_head", True)),
        )

    if name == "ttfields_unetr":
        return TTFieldsUNETR(
            in_channels=int(model_cfg.get("in_channels", 4)),
            out_channels=int(model_cfg.get("out_channels", 4)),
            img_size=int(model_cfg.get("img_size", 128)),
            feature_size=int(model_cfg.get("feature_size", 16)),
            hidden_size=int(model_cfg.get("hidden_size", 768)),
            mlp_dim=int(model_cfg.get("mlp_dim", 3072)),
            num_heads=int(model_cfg.get("num_heads", 12)),
            dropout_rate=float(model_cfg.get("dropout_rate", 0.0)),
            use_coord_head=bool(model_cfg.get("use_coord_head", False)),
        )

    raise ValueError(f"Unsupported model name: {name}")


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
    name = str(cfg["training"].get("scheduler", "none"))
    if name == "cosine":
        lr_min = float(cfg["training"].get("lr_min", 1e-6))
        return torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=num_epochs, eta_min=lr_min
        )
    return None
