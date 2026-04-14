from __future__ import annotations

import contextlib
import json
import math
from pathlib import Path

import torch
from torch.optim import AdamW, SGD
from torch.utils.data import DataLoader

from ..io import ensure_parent
from .common import checkpoint_path, resolve_device, set_random_seed
from .config import experiment_output_dir, load_config
from .dataset import SupConVideoDataset
from .losses import SupConLoss
from .model import build_supcon_model


def run_supcon_training(config_path: str | Path) -> Path:
    config = load_config(config_path)
    set_random_seed(int(config["experiment"].get("seed", 7)))
    training_config = config["training"]
    device = resolve_device(training_config.get("device"))
    output_dir = experiment_output_dir(config)
    dataset_config = config["dataset"]
    input_size = (int(dataset_config.get("input_width", 224)), int(dataset_config.get("input_height", 224)))

    dataset = SupConVideoDataset(
        index_path=dataset_config["video_index"],
        input_size=input_size,
        clip_length=int(dataset_config.get("clip_length", 8)),
        frame_stride=int(dataset_config.get("frame_stride", 2)),
        temporal_jitter=int(dataset_config.get("temporal_jitter", 1)),
        crop_scale_range=(
            float(dataset_config.get("crop_scale_min", 0.7)),
            float(dataset_config.get("crop_scale_max", 1.0)),
        ),
        color_jitter_strength=float(dataset_config.get("color_jitter_strength", 0.4)),
        grayscale_prob=float(dataset_config.get("grayscale_prob", 0.2)),
        blur_prob=float(dataset_config.get("blur_prob", 0.5)),
        blur_kernel_size=int(dataset_config.get("blur_kernel_size", 5)),
    )
    loader = DataLoader(
        dataset,
        batch_size=int(training_config.get("batch_size", 8)),
        shuffle=True,
        num_workers=int(training_config.get("num_workers", 0)),
        pin_memory=device == "cuda",
    )

    model = build_supcon_model(config).to(device)
    optimizer = _build_optimizer(model, training_config)
    loss_fn = SupConLoss(temperature=float(training_config.get("temperature", 0.07)))
    epochs = int(training_config.get("epochs", 10))
    accumulation_steps = max(int(training_config.get("gradient_accumulation_steps", 1)), 1)
    clip_grad_norm = max(float(training_config.get("clip_grad_norm", 0.0)), 0.0)
    warmup_epochs = max(int(training_config.get("warmup_epochs", 0)), 0)
    amp_enabled = bool(training_config.get("amp", False)) and device == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=amp_enabled)
    base_lrs = [float(group["lr"]) for group in optimizer.param_groups]

    epoch_logs: list[dict[str, float]] = []
    model.train()
    for epoch in range(epochs):
        current_lr = _apply_epoch_learning_rate(
            optimizer=optimizer,
            base_lrs=base_lrs,
            epoch=epoch,
            epochs=epochs,
            use_cosine_schedule=bool(training_config.get("use_cosine_schedule", True)),
            warmup_epochs=warmup_epochs,
        )
        running_loss = 0.0
        optimizer.zero_grad(set_to_none=True)
        for batch_index, batch in enumerate(loader, start=1):
            view_1 = batch["view_1"].to(device)
            view_2 = batch["view_2"].to(device)
            labels = batch["label"].to(device)

            clips = torch.cat([view_1, view_2], dim=0)
            expanded_labels = torch.cat([labels, labels], dim=0)
            with _autocast_context(device=device, enabled=amp_enabled):
                outputs = model(clips)
                raw_loss = loss_fn(outputs["projections"], expanded_labels)
            running_loss += float(raw_loss.detach().cpu())
            loss = raw_loss / accumulation_steps

            if amp_enabled:
                scaler.scale(loss).backward()
            else:
                loss.backward()

            should_step = batch_index % accumulation_steps == 0 or batch_index == len(loader)
            if should_step:
                if clip_grad_norm > 0:
                    if amp_enabled:
                        scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=clip_grad_norm)
                if amp_enabled:
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    optimizer.step()
                optimizer.zero_grad(set_to_none=True)

        average_loss = running_loss / max(len(loader), 1)
        epoch_logs.append(
            {
                "epoch": epoch + 1,
                "loss": average_loss,
                "learning_rate": current_lr,
            }
        )

    checkpoint = checkpoint_path(output_dir)
    torch.save(
        {
            "checkpoint_type": "supcon_video_pretraining",
            "encoder_backbone": model.encoder.backbone_name,
            "reuse_targets": ["video"],
            "model": model.state_dict(),
            "encoder": model.encoder.state_dict(),
            "projection_head": model.projection_head.state_dict(),
            "label_to_index": dataset.label_to_index,
            "config": config,
        },
        checkpoint,
    )
    metrics_path = ensure_parent(output_dir / "train_metrics.json")
    with metrics_path.open("w", encoding="utf-8") as handle:
        json.dump(epoch_logs, handle, indent=2)
        handle.write("\n")
    return checkpoint


def _build_optimizer(model: torch.nn.Module, training_config: dict) -> torch.optim.Optimizer:
    optimizer_name = str(training_config.get("optimizer", "sgd")).lower()
    learning_rate = float(training_config.get("learning_rate", 1e-2))
    weight_decay = float(training_config.get("weight_decay", 1e-4))
    if optimizer_name == "adamw":
        return AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    momentum = float(training_config.get("momentum", 0.9))
    return SGD(model.parameters(), lr=learning_rate, weight_decay=weight_decay, momentum=momentum)


def _apply_epoch_learning_rate(
    optimizer: torch.optim.Optimizer,
    base_lrs: list[float],
    epoch: int,
    epochs: int,
    use_cosine_schedule: bool,
    warmup_epochs: int,
) -> float:
    scale = _learning_rate_scale(
        epoch=epoch,
        epochs=epochs,
        use_cosine_schedule=use_cosine_schedule,
        warmup_epochs=warmup_epochs,
    )
    current_lr = base_lrs[0] * scale
    for param_group, base_lr in zip(optimizer.param_groups, base_lrs):
        param_group["lr"] = base_lr * scale
    return current_lr


def _learning_rate_scale(epoch: int, epochs: int, use_cosine_schedule: bool, warmup_epochs: int) -> float:
    if warmup_epochs > 0 and epoch < warmup_epochs:
        return float(epoch + 1) / float(max(warmup_epochs, 1))
    if not use_cosine_schedule or epochs <= warmup_epochs + 1:
        return 1.0
    cosine_epoch = epoch - warmup_epochs
    cosine_total = max(epochs - warmup_epochs - 1, 1)
    progress = min(max(cosine_epoch / cosine_total, 0.0), 1.0)
    return 0.5 * (1.0 + math.cos(math.pi * progress))


def _autocast_context(device: str, enabled: bool) -> contextlib.AbstractContextManager:
    if enabled and device == "cuda":
        return torch.autocast(device_type="cuda", enabled=True)
    return contextlib.nullcontext()
