from __future__ import annotations

import json
from pathlib import Path

import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader

from ..constants import KEYPOINT_NAMES
from ..io import ensure_parent
from .common import checkpoint_path, resolve_device, set_random_seed
from .config import experiment_output_dir, load_config
from .dataset import PoseDataset
from .losses import supervised_pose_loss
from .model import build_model


def run_supervised_training(config_path: str | Path) -> Path:
    config = load_config(config_path)
    set_random_seed(int(config["experiment"].get("seed", 7)))
    device = resolve_device()
    output_dir = experiment_output_dir(config)
    dataset_config = config["dataset"]
    training_config = config["training"]
    input_size = (dataset_config["input_width"], dataset_config["input_height"])
    heatmap_size = (dataset_config["heatmap_width"], dataset_config["heatmap_height"])
    index_path = dataset_config.get("train_index") or dataset_config.get("annotation_index")
    if not index_path:
        raise ValueError("dataset.train_index or dataset.annotation_index must be configured")

    dataset = PoseDataset(
        index_path=index_path,
        image_root=dataset_config["image_root"],
        input_size=input_size,
        heatmap_size=heatmap_size,
    )
    loader = DataLoader(
        dataset,
        batch_size=int(training_config.get("batch_size", 8)),
        shuffle=True,
        num_workers=int(training_config.get("num_workers", 0)),
    )

    model = build_model(config, len(KEYPOINT_NAMES)).to(device)
    optimizer = AdamW(
        model.parameters(),
        lr=float(training_config.get("learning_rate", 5e-4)),
        weight_decay=float(training_config.get("weight_decay", 1e-4)),
    )
    visibility_loss_weight = float(training_config.get("visibility_loss_weight", 0.2))
    epoch_logs: list[dict[str, float]] = []

    model.train()
    for epoch in range(int(training_config.get("epochs", 10))):
        running_loss = 0.0
        for batch in loader:
            images = batch["image"].to(device)
            heatmaps = batch["heatmaps"].to(device)
            visibility = batch["visibility"].to(device)
            predictions = model(images)
            loss, components = supervised_pose_loss(
                predictions=predictions,
                target_heatmaps=heatmaps,
                target_visibility=visibility,
                visibility_loss_weight=visibility_loss_weight,
            )
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            running_loss += float(loss.detach().cpu())
        average_loss = running_loss / max(len(loader), 1)
        epoch_logs.append({"epoch": epoch + 1, "loss": average_loss})

    checkpoint = checkpoint_path(output_dir)
    torch.save({"model": model.state_dict(), "config": config}, checkpoint)
    metrics_path = ensure_parent(output_dir / "train_metrics.json")
    with metrics_path.open("w", encoding="utf-8") as handle:
        json.dump(epoch_logs, handle, indent=2)
        handle.write("\n")
    return checkpoint

