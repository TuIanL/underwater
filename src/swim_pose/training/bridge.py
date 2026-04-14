from __future__ import annotations

from pathlib import Path

import torch
import torch.nn.functional as F
from torch import nn

from .model import VideoBackboneEncoder


class FeatureBridgeProjector(nn.Module):
    def __init__(self, input_dim: int, output_dim: int) -> None:
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_dim, output_dim),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.layers(features)


def bridge_feature_loss(student_features: torch.Tensor, teacher_features: torch.Tensor) -> torch.Tensor:
    if student_features.shape != teacher_features.shape:
        raise ValueError(
            "Bridge feature loss requires student and teacher features with the same shape, "
            f"got {tuple(student_features.shape)} and {tuple(teacher_features.shape)}"
        )
    student_normalized = F.normalize(student_features, dim=1)
    teacher_normalized = F.normalize(teacher_features, dim=1)
    return F.mse_loss(student_normalized, teacher_normalized)


def load_bridge_teacher(checkpoint_path: str | Path) -> tuple[VideoBackboneEncoder, dict[str, str]]:
    path = Path(checkpoint_path)
    state = torch.load(path, map_location="cpu")
    if not isinstance(state, dict):
        raise ValueError("Bridge teacher checkpoint must be a dict-backed SupCon checkpoint.")
    if state.get("checkpoint_type") != "supcon_video_pretraining":
        raise ValueError("Bridge teacher checkpoint must come from Phase 1 SupCon video pretraining.")
    if "encoder" not in state:
        raise ValueError("Bridge teacher checkpoint is missing encoder weights.")

    backbone = str(
        state.get("encoder_backbone")
        or state.get("config", {}).get("model", {}).get("backbone", "r2plus1d_18")
    )
    teacher = VideoBackboneEncoder(backbone=backbone, pretrained_backbone=False)
    teacher.load_state_dict(state["encoder"], strict=False)
    teacher.eval()
    for parameter in teacher.parameters():
        parameter.requires_grad = False
    metadata = {
        "checkpoint_type": str(state.get("checkpoint_type", "")),
        "backbone": backbone,
        "checkpoint_path": str(path),
    }
    return teacher, metadata
