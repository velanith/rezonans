from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader

from training.builder import build_datasets, build_loaders, build_loss, build_model, build_scheduler
from training.losses import MultiTaskLoss
from training.metrics import coord_mae_norm, dice_wt_tc_et
from training.utils import autocast_context, move_batch_to_device, resolve_device, save_checkpoint, set_seed


@torch.no_grad()
def compute_metrics(
    model_out: dict[str, torch.Tensor],
    batch: dict[str, torch.Tensor],
) -> dict[str, float]:
    pred_mask = torch.argmax(model_out["seg_logits"], dim=1)
    batch_size = pred_mask.shape[0]

    dice_wt_sum = dice_tc_sum = dice_et_sum = 0.0
    for i in range(batch_size):
        d = dice_wt_tc_et(
            pred_mask=pred_mask[i].detach().cpu(),
            target_mask=batch["seg"][i].detach().cpu(),
        )
        dice_wt_sum += float(d["dice_wt"].item())
        dice_tc_sum += float(d["dice_tc"].item())
        dice_et_sum += float(d["dice_et"].item())

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
    sums: dict[str, float] = {
        "dice_wt": 0.0, "dice_tc": 0.0, "dice_et": 0.0,
        "coord_wt_mae_norm": 0.0, "coord_et_mae_norm": 0.0,
    }

    for batch in loader:
        batch = move_batch_to_device(batch, device)
        with autocast_context(device, precision):
            model_out = model(batch["image"])
            loss_terms = criterion(model_out, batch, epoch=epoch)

        total_loss += float(loss_terms["loss_total"].detach().cpu().item())
        num_batches += 1
        for k, v in compute_metrics(model_out, batch).items():
            sums[k] += v

    n = max(1, num_batches)
    return {"val_loss": total_loss / n, **{f"val_{k}": v / n for k, v in sums.items()}}


def train(cfg: dict[str, Any]) -> None:
    run_cfg = cfg["run"]
    training_cfg = cfg["training"]

    set_seed(int(run_cfg.get("seed", 42)))
    device = resolve_device(str(training_cfg.get("device", "auto")))
    precision = str(training_cfg.get("precision", "fp32"))

    print(f"Run: {run_cfg.get('name', 'unnamed')}")
    print(f"Device: {device} | Precision: {precision}")

    train_ds, val_ds = build_datasets(cfg)
    train_loader, val_loader = build_loaders(train_ds, val_ds, cfg, device)

    model = build_model(cfg).to(device)

    pretrained = training_cfg.get("pretrained", None)
    if pretrained:
        ckpt = torch.load(pretrained, map_location=device, weights_only=True)
        missing, unexpected = model.load_state_dict(ckpt["model_state_dict"], strict=False)
        print(f"Loaded pretrained: {pretrained} (epoch {ckpt.get('epoch', '?')})")
        if missing:
            print(f"  New params (random init): {len(missing)}")

    if bool(training_cfg.get("freeze_backbone", False)):
        for name, param in model.named_parameters():
            if not name.startswith("coord_head"):
                param.requires_grad = False
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        total = sum(p.numel() for p in model.parameters())
        print(f"Backbone frozen. Trainable: {trainable:,} / {total:,} params")

    criterion = build_loss(cfg)
    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
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

    start_epoch = 0
    resume_path = training_cfg.get("resume", None)
    if resume_path:
        ckpt = torch.load(resume_path, map_location=device, weights_only=True)
        model.load_state_dict(ckpt["model_state_dict"])
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        start_epoch = int(ckpt.get("epoch", 0)) + 1
        for _ in range(start_epoch):
            scheduler.step()
        print(f"Resumed from {resume_path} (epoch {start_epoch - 1} → continuing from {start_epoch})")

    default_key = "val_dice_wt" if val_loader is not None else "dice_wt"
    best_key = str(training_cfg.get("monitor_metric", default_key))
    # For MAE metrics lower is better; negate so the same > comparison works.
    monitor_sign = -1.0 if "mae" in best_key else 1.0
    best_score = -float("inf")

    for epoch in range(start_epoch, max_epochs):
        model.train()
        t0 = time.perf_counter()

        epoch_loss = 0.0
        num_batches = 0
        term_sums: dict[str, float] = {
            "loss_seg": 0.0, "loss_coord": 0.0,
            "loss_coord_wt": 0.0, "loss_coord_et": 0.0, "lambda_coord": 0.0,
        }
        metric_sums: dict[str, float] = {
            "dice_wt": 0.0, "dice_tc": 0.0, "dice_et": 0.0,
            "coord_wt_mae_norm": 0.0, "coord_et_mae_norm": 0.0,
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
            for k, v in compute_metrics(model_out, batch).items():
                metric_sums[k] += v

        if scheduler is not None:
            scheduler.step()

        n = max(1, num_batches)
        avg_loss = epoch_loss / n
        avg_terms = {k: v / n for k, v in term_sums.items()}
        avg_metrics = {k: v / n for k, v in metric_sums.items()}
        elapsed = time.perf_counter() - t0
        lr = optimizer.param_groups[0]["lr"]

        val_metrics: dict[str, float] = {}
        if val_loader is not None:
            val_metrics = validate(model, val_loader, criterion, device, precision, epoch)

        if epoch % log_interval == 0:
            line = (
                f"epoch={epoch:03d} loss={avg_loss:.4f} "
                f"seg={avg_terms['loss_seg']:.4f} coord={avg_terms['loss_coord']:.4f} "
                f"λ={avg_terms['lambda_coord']:.4f} "
                f"dice_wt={avg_metrics['dice_wt']:.4f} "
                f"dice_tc={avg_metrics['dice_tc']:.4f} "
                f"dice_et={avg_metrics['dice_et']:.4f} "
                f"wt_mae={avg_metrics['coord_wt_mae_norm']:.4f} "
                f"et_mae={avg_metrics['coord_et_mae_norm']:.4f} "
                f"lr={lr:.2e} t={elapsed:.1f}s"
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
            monitor = monitor_sign * all_metrics.get(best_key, 0.0)
            save_checkpoint(model, optimizer, epoch, all_metrics, ckpt_dir / "last.pt")
            if monitor > best_score:
                best_score = monitor
                save_checkpoint(model, optimizer, epoch, all_metrics, ckpt_dir / "best.pt")
                display = all_metrics.get(best_key, 0.0)
                print(f"  → best saved (epoch={epoch}, {best_key}={display:.4f})")
