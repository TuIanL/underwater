from __future__ import annotations

from pathlib import Path

import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader

from ..constants import KEYPOINT_NAMES
from .common import checkpoint_path, resolve_device, set_random_seed
from .config import experiment_output_dir, load_config
from .dataset import PoseDataset, TemporalUnlabeledFrameDataset, UnlabeledFrameDataset
from .losses import consistency_loss, supervised_pose_loss, temporal_smoothness_loss
from .model import build_model


def run_semi_supervised_training(config_path: str | Path) -> Path:
    config = load_config(config_path)
    set_random_seed(int(config["experiment"].get("seed", 7)))
    device = resolve_device()
    output_dir = experiment_output_dir(config)
    dataset_config = config["dataset"]
    training_config = config["training"]
    input_size = (dataset_config["input_width"], dataset_config["input_height"])
    heatmap_size = (dataset_config["heatmap_width"], dataset_config["heatmap_height"])

    temporal_weight = float(training_config.get("temporal_loss_weight", 0.0))
    temporal_threshold = float(training_config.get("temporal_threshold", training_config.get("pseudolabel_threshold", 0.5)))
    unlabeled_dataset_cls = TemporalUnlabeledFrameDataset if temporal_weight > 0 else UnlabeledFrameDataset

    labeled_dataset = PoseDataset(
        index_path=dataset_config["labeled_index"],
        image_root=dataset_config["image_root"],
        input_size=input_size,
        heatmap_size=heatmap_size,
    )
    unlabeled_dataset = unlabeled_dataset_cls(
        index_path=dataset_config["unlabeled_index"],
        image_root=dataset_config["image_root"],
        input_size=input_size,
    )
    labeled_loader = DataLoader(
        labeled_dataset,
        batch_size=int(training_config.get("batch_size", 8)),
        shuffle=True,
        num_workers=int(training_config.get("num_workers", 0)),
    )
    unlabeled_loader = DataLoader(
        unlabeled_dataset,
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
    consistency_weight = float(training_config.get("consistency_loss_weight", 0.5))
    threshold = float(training_config.get("pseudolabel_threshold", 0.5))

    model.train()
    unlabeled_iterator = iter(unlabeled_loader)
    for _ in range(int(training_config.get("epochs", 10))):
        for labeled_batch in labeled_loader:
            try:
                unlabeled_batch = next(unlabeled_iterator)
            except StopIteration:
                unlabeled_iterator = iter(unlabeled_loader)
                unlabeled_batch = next(unlabeled_iterator)

            labeled_images = labeled_batch["image"].to(device)
            labeled_heatmaps = labeled_batch["heatmaps"].to(device)
            labeled_visibility = labeled_batch["visibility"].to(device)
            supervised_predictions = model(labeled_images)
            supervised_loss, _ = supervised_pose_loss(
                predictions=supervised_predictions,
                target_heatmaps=labeled_heatmaps,
                target_visibility=labeled_visibility,
                visibility_loss_weight=visibility_loss_weight,
            )

            unlabeled_images = unlabeled_batch["image"].to(device)
            weak_predictions = model(unlabeled_images)
            strong_predictions = model(_strong_augment(unlabeled_images))
            unsupervised_loss = consistency_loss(weak_predictions, strong_predictions, threshold=threshold)
            temporal_loss = weak_predictions["heatmaps"].new_tensor(0.0)
            if temporal_weight > 0:
                temporal_images = unlabeled_batch["temporal_image"].to(device)
                temporal_predictions = model(temporal_images)
                temporal_loss = temporal_smoothness_loss(
                    current_predictions=weak_predictions,
                    temporal_predictions=temporal_predictions,
                    has_temporal_pair=unlabeled_batch["has_temporal_pair"].to(device),
                    threshold=temporal_threshold,
                )

            loss = supervised_loss + consistency_weight * unsupervised_loss + temporal_weight * temporal_loss
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    checkpoint = checkpoint_path(output_dir)
    torch.save({"model": model.state_dict(), "config": config}, checkpoint)
    return checkpoint


def _strong_augment(images: torch.Tensor) -> torch.Tensor:
    noise = torch.randn_like(images) * 0.05
    return torch.clamp(images + noise, 0.0, 1.0)
