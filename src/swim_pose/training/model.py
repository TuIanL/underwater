from __future__ import annotations

from pathlib import Path

import torch
from torch import nn
from torchvision.models import ResNet18_Weights, ResNet34_Weights, resnet18, resnet34
from torchvision.models.video import (
    MC3_18_Weights,
    R2Plus1D_18_Weights,
    R3D_18_Weights,
    mc3_18,
    r2plus1d_18,
    r3d_18,
)


def _load_resnet_backbone(backbone: str, pretrained_backbone: bool) -> tuple[nn.Module, int]:
    if backbone == "resnet34":
        weights = ResNet34_Weights.DEFAULT if pretrained_backbone else None
        return resnet34(weights=weights), 512
    weights = ResNet18_Weights.DEFAULT if pretrained_backbone else None
    return resnet18(weights=weights), 512


def _load_video_backbone(backbone: str, pretrained_backbone: bool) -> tuple[nn.Module, int]:
    builders = {
        "r2plus1d_18": (r2plus1d_18, R2Plus1D_18_Weights),
        "r3d_18": (r3d_18, R3D_18_Weights),
        "mc3_18": (mc3_18, MC3_18_Weights),
    }
    if backbone not in builders:
        supported = ", ".join(sorted(builders))
        raise ValueError(f"Unsupported SupCon video backbone '{backbone}'. Expected one of: {supported}")
    builder, weights_enum = builders[backbone]
    weights = weights_enum.DEFAULT if pretrained_backbone else None
    model = builder(weights=weights)
    return model, int(model.fc.in_features)


class ResNetHeatmapModel(nn.Module):
    def __init__(self, num_keypoints: int, backbone: str = "resnet18", pretrained_backbone: bool = True) -> None:
        super().__init__()
        backbone_model, feature_dim = _load_resnet_backbone(backbone, pretrained_backbone)

        self.stem = nn.Sequential(
            backbone_model.conv1,
            backbone_model.bn1,
            backbone_model.relu,
            backbone_model.maxpool,
            backbone_model.layer1,
            backbone_model.layer2,
            backbone_model.layer3,
            backbone_model.layer4,
        )
        self.deconv = nn.Sequential(
            nn.ConvTranspose2d(512, 256, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )
        self.heatmap_head = nn.Conv2d(64, num_keypoints, kernel_size=1)
        self.visibility_head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(feature_dim, num_keypoints * 3),
        )
        self.num_keypoints = num_keypoints
        self.feature_dim = feature_dim

    def forward(self, image: torch.Tensor) -> dict[str, torch.Tensor]:
        features = self.stem(image)
        pooled_features = nn.functional.adaptive_avg_pool2d(features, output_size=1).flatten(1)
        heatmaps = self.heatmap_head(self.deconv(features))
        visibility_logits = self.visibility_head(features).view(image.shape[0], self.num_keypoints, 3)
        return {
            "heatmaps": heatmaps,
            "visibility_logits": visibility_logits,
            "pooled_features": pooled_features,
        }


class VideoBackboneEncoder(nn.Module):
    def __init__(self, backbone: str = "r2plus1d_18", pretrained_backbone: bool = True) -> None:
        super().__init__()
        backbone_model, feature_dim = _load_video_backbone(backbone, pretrained_backbone)
        self.backbone_name = backbone
        self.feature_extractor = nn.Sequential(
            backbone_model.stem,
            backbone_model.layer1,
            backbone_model.layer2,
            backbone_model.layer3,
            backbone_model.layer4,
        )
        self.pool = nn.AdaptiveAvgPool3d(1)
        self.feature_dim = feature_dim

    def forward(self, clips: torch.Tensor) -> torch.Tensor:
        if clips.ndim != 5:
            raise ValueError(f"Expected clips shaped BxCxTxHxW, got {tuple(clips.shape)}")
        encoded = self.feature_extractor(clips)
        return self.pool(encoded).flatten(1)


class ProjectionHead(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int) -> None:
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        projections = self.layers(features)
        return nn.functional.normalize(projections, dim=1)


class SupConPretrainingModel(nn.Module):
    def __init__(
        self,
        backbone: str = "r2plus1d_18",
        pretrained_backbone: bool = True,
        projection_hidden_dim: int | None = None,
        projection_dim: int = 128,
    ) -> None:
        super().__init__()
        self.encoder = VideoBackboneEncoder(backbone=backbone, pretrained_backbone=pretrained_backbone)
        hidden_dim = projection_hidden_dim or self.encoder.feature_dim
        self.projection_head = ProjectionHead(
            input_dim=self.encoder.feature_dim,
            hidden_dim=hidden_dim,
            output_dim=projection_dim,
        )

    def forward(self, clips: torch.Tensor) -> dict[str, torch.Tensor]:
        features = self.encoder(clips)
        projections = self.projection_head(features)
        return {"features": features, "projections": projections}

    def encode(self, clips: torch.Tensor) -> torch.Tensor:
        return self.encoder(clips)


def build_model(config: dict, num_keypoints: int) -> nn.Module:
    model = ResNetHeatmapModel(
        num_keypoints=num_keypoints,
        backbone=config["model"].get("backbone", "resnet18"),
        pretrained_backbone=bool(config["model"].get("pretrained_backbone", True)),
    )
    checkpoint_path = config["model"].get("pretrained_checkpoint", "")
    if checkpoint_path and Path(checkpoint_path).exists():
        state = torch.load(checkpoint_path, map_location="cpu")
        if _is_supcon_pretraining_checkpoint(state):
            raise ValueError(
                "SupCon video pretraining checkpoints are not directly compatible with the 2D heatmap model. "
                "Use the video encoder checkpoint with a compatible downstream video model instead."
            )
        state_dict = state["model"] if isinstance(state, dict) and "model" in state else state
        model.load_state_dict(state_dict, strict=False)
    return model


def build_supcon_model(config: dict) -> SupConPretrainingModel:
    model_config = config["model"]
    model = SupConPretrainingModel(
        backbone=model_config.get("backbone", "r2plus1d_18"),
        pretrained_backbone=bool(model_config.get("pretrained_backbone", True)),
        projection_hidden_dim=int(model_config.get("projection_hidden_dim", 512)),
        projection_dim=int(model_config.get("projection_dim", 128)),
    )
    checkpoint_path = model_config.get("pretrained_checkpoint", "")
    if checkpoint_path and Path(checkpoint_path).exists():
        state = torch.load(checkpoint_path, map_location="cpu")
        if isinstance(state, dict) and "encoder" in state:
            model.encoder.load_state_dict(state["encoder"], strict=False)
            if "projection_head" in state:
                model.projection_head.load_state_dict(state["projection_head"], strict=False)
        else:
            state_dict = state["model"] if isinstance(state, dict) and "model" in state else state
            model.load_state_dict(state_dict, strict=False)
    return model


def _is_supcon_pretraining_checkpoint(state: object) -> bool:
    return isinstance(state, dict) and "encoder" in state and "projection_head" in state
