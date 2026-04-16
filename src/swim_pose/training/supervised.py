from __future__ import annotations

import json
from pathlib import Path

import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader

from ..constants import KEYPOINT_NAMES
from ..io import ensure_parent
from ..pathing import serialize_workspace_path
from .bridge import FeatureBridgeProjector, bridge_feature_loss, load_bridge_teacher
from .common import checkpoint_path, forward_with_singleton_batch_support, resolve_device, set_random_seed
from .config import experiment_output_dir, load_config
from .dataset import PoseDataset, TemporalPoseDataset
from .losses import supervised_pose_loss
from .model import build_model


def run_supervised_training(config_path: str | Path) -> Path:
    config = load_config(config_path)
    set_random_seed(int(config["experiment"].get("seed", 7)))
    training_config = config["training"]
    device = resolve_device(training_config.get("device"))
    output_dir = experiment_output_dir(config)
    dataset_config = config["dataset"]
    bridge_config = config.get("bridge", {})
    input_size = (dataset_config["input_width"], dataset_config["input_height"])
    heatmap_size = (dataset_config["heatmap_width"], dataset_config["heatmap_height"])
    index_path = dataset_config.get("train_index") or dataset_config.get("annotation_index")
    if not index_path:
        raise ValueError("dataset.train_index or dataset.annotation_index must be configured")
    bridge_enabled = bool(bridge_config.get("enabled", False))

    if bridge_enabled:
        bridge_input_size = (
            int(bridge_config.get("input_width", input_size[0])),
            int(bridge_config.get("input_height", input_size[1])),
        )
        bridge_context_index = bridge_config.get("context_index", "")
        if not bridge_context_index:
            raise ValueError("bridge.context_index must be configured when bridge.enabled is true")
        dataset = TemporalPoseDataset(
            index_path=index_path,
            image_root=dataset_config["image_root"],
            input_size=input_size,
            heatmap_size=heatmap_size,
            bridge_input_size=bridge_input_size,
            bridge_clip_length=int(bridge_config.get("clip_length", 8)),
            bridge_frame_stride=int(bridge_config.get("frame_stride", 1)),
            bridge_context_index=bridge_context_index,
        )
    else:
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
    teacher = None
    bridge_projector = None
    bridge_weight = 0.0
    bridge_skip_missing_context = bool(bridge_config.get("skip_missing_context", True))
    optimizer_parameters = list(model.parameters())
    bridge_metadata: dict[str, object] = {}
    if bridge_enabled:
        teacher_checkpoint = bridge_config.get("teacher_checkpoint", "")
        if not teacher_checkpoint:
            raise ValueError("bridge.teacher_checkpoint must be configured when bridge.enabled is true")
        teacher, teacher_metadata = load_bridge_teacher(teacher_checkpoint)
        teacher = teacher.to(device)
        bridge_projector = FeatureBridgeProjector(model.feature_dim, teacher.feature_dim).to(device)
        optimizer_parameters.extend(bridge_projector.parameters())
        bridge_weight = float(bridge_config.get("distillation_weight", 0.1))
        bridge_metadata = {
            "bridge_type": "video_feature_distillation",
            "bridge_teacher_checkpoint": serialize_workspace_path(teacher_checkpoint),
            "bridge_teacher_backbone": teacher_metadata["backbone"],
            "bridge_context_index": serialize_workspace_path(bridge_context_index),
            "bridge_skip_missing_context": bridge_skip_missing_context,
        }

    optimizer = AdamW(
        optimizer_parameters,
        lr=float(training_config.get("learning_rate", 5e-4)),
        weight_decay=float(training_config.get("weight_decay", 1e-4)),
    )
    visibility_loss_weight = float(training_config.get("visibility_loss_weight", 0.2))
    epoch_logs: list[dict[str, float]] = []

    model.train()
    for epoch in range(int(training_config.get("epochs", 10))):
        running_loss = 0.0
        running_bridge_loss = 0.0
        bridge_context_samples = 0
        bridge_total_samples = 0
        for batch in loader:
            images = batch["image"].to(device)
            heatmaps = batch["heatmaps"].to(device)
            visibility = batch["visibility"].to(device)
            predictions = forward_with_singleton_batch_support(model, images)
            supervised_total, components = supervised_pose_loss(
                predictions=predictions,
                target_heatmaps=heatmaps,
                target_visibility=visibility,
                visibility_loss_weight=visibility_loss_weight,
            )
            loss = supervised_total
            if bridge_enabled and teacher is not None and bridge_projector is not None:
                has_bridge_context = batch["has_bridge_context"].bool()
                bridge_total_samples += int(has_bridge_context.numel())
                bridge_context_samples += int(has_bridge_context.sum().item())
                if not bridge_skip_missing_context and not torch.all(has_bridge_context):
                    raise ValueError("Dense bridge context is missing for one or more samples in the batch.")

                if torch.any(has_bridge_context):
                    valid_mask = has_bridge_context.to(device=device)
                    with torch.no_grad():
                        teacher_features = teacher(batch["bridge_clip"][has_bridge_context].to(device))
                    student_features = bridge_projector(predictions["pooled_features"][valid_mask])
                    distillation_loss = bridge_feature_loss(student_features, teacher_features)
                    loss = loss + bridge_weight * distillation_loss
                    components["bridge_loss"] = float(distillation_loss.detach().cpu())
                    running_bridge_loss += components["bridge_loss"]
                else:
                    components["bridge_loss"] = 0.0
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            running_loss += float(loss.detach().cpu())
        average_loss = running_loss / max(len(loader), 1)
        epoch_log = {"epoch": epoch + 1, "loss": average_loss}
        if bridge_enabled:
            epoch_log["bridge_loss"] = running_bridge_loss / max(len(loader), 1)
            epoch_log["bridge_context_coverage"] = bridge_context_samples / max(bridge_total_samples, 1)
        epoch_logs.append(epoch_log)

    checkpoint = checkpoint_path(output_dir)
    payload = {
        "checkpoint_type": "localization_bridge" if bridge_enabled else "localization",
        "training_mode": "supervised",
        "prediction_contract": "frame_keyed",
        "conservative_upgrade": {
            "stage": str(config["experiment"].get("conservative_stage", "stage0_frame_only")),
            "bridge": {
                "enabled": bridge_enabled,
                "skip_missing_context": bridge_skip_missing_context,
            },
        },
        "model": model.state_dict(),
        "config": config,
    }
    if bridge_projector is not None:
        payload["bridge_projector"] = bridge_projector.state_dict()
    payload.update(bridge_metadata)
    torch.save(payload, checkpoint)
    metrics_path = ensure_parent(output_dir / "train_metrics.json")
    with metrics_path.open("w", encoding="utf-8") as handle:
        json.dump(epoch_logs, handle, indent=2)
        handle.write("\n")
    return checkpoint
