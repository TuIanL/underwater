from __future__ import annotations

import json
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
    model_config = config["model"]
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
    )

    model = build_supcon_model(config).to(device)
    optimizer = _build_optimizer(model, training_config)
    scheduler = _build_scheduler(optimizer, training_config)
    loss_fn = SupConLoss(temperature=float(training_config.get("temperature", 0.07)))

    epoch_logs: list[dict[str, float]] = []
    epochs = int(training_config.get("epochs", 10))
    model.train()
    for epoch in range(epochs):
        running_loss = 0.0
        for batch in loader:
            view_1 = batch["view_1"].to(device)
            view_2 = batch["view_2"].to(device)
            labels = batch["label"].to(device)

            clips = torch.cat([view_1, view_2], dim=0)
            expanded_labels = torch.cat([labels, labels], dim=0)
            outputs = model(clips)
            loss = loss_fn(outputs["projections"], expanded_labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            running_loss += float(loss.detach().cpu())

        if scheduler is not None:
            scheduler.step()

        average_loss = running_loss / max(len(loader), 1)
        epoch_logs.append(
            {
                "epoch": epoch + 1,
                "loss": average_loss,
                "learning_rate": float(optimizer.param_groups[0]["lr"]),
            }
        )

    checkpoint = checkpoint_path(output_dir)
    torch.save(
        {
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


def _build_scheduler(
    optimizer: torch.optim.Optimizer,
    training_config: dict,
) -> torch.optim.lr_scheduler._LRScheduler | torch.optim.lr_scheduler.LRScheduler | None:
    if not bool(training_config.get("use_cosine_schedule", True)):
        return None
    epochs = int(training_config.get("epochs", 10))
    if epochs <= 1:
        return None
    return torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
