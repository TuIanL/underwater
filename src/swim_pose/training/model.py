from __future__ import annotations

from pathlib import Path

import torch
from torch import nn
from torchvision.models import ResNet18_Weights, ResNet34_Weights, resnet18, resnet34


def _load_resnet_backbone(backbone: str, pretrained_backbone: bool) -> tuple[nn.Module, int]:
    if backbone == "resnet34":
        weights = ResNet34_Weights.DEFAULT if pretrained_backbone else None
        return resnet34(weights=weights), 512
    weights = ResNet18_Weights.DEFAULT if pretrained_backbone else None
    return resnet18(weights=weights), 512


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

    def forward(self, image: torch.Tensor) -> dict[str, torch.Tensor]:
        features = self.stem(image)
        heatmaps = self.heatmap_head(self.deconv(features))
        visibility_logits = self.visibility_head(features).view(image.shape[0], self.num_keypoints, 3)
        return {
            "heatmaps": heatmaps,
            "visibility_logits": visibility_logits,
        }


class TemporalResNetEncoder(nn.Module):
    def __init__(self, backbone: str = "resnet18", pretrained_backbone: bool = True) -> None:
        super().__init__()
        backbone_model, feature_dim = _load_resnet_backbone(backbone, pretrained_backbone)
        self.frame_encoder = nn.Sequential(
            backbone_model.conv1,
            backbone_model.bn1,
            backbone_model.relu,
            backbone_model.maxpool,
            backbone_model.layer1,
            backbone_model.layer2,
            backbone_model.layer3,
            backbone_model.layer4,
        )
        self.feature_dim = feature_dim

    def forward(self, clips: torch.Tensor) -> torch.Tensor:
        if clips.ndim != 5:
            raise ValueError(f"Expected clips shaped BxCxTxHxW, got {tuple(clips.shape)}")
        batch_size, channels, frames, height, width = clips.shape
        flattened = clips.permute(0, 2, 1, 3, 4).reshape(batch_size * frames, channels, height, width)
        encoded = self.frame_encoder(flattened)
        pooled = nn.functional.adaptive_avg_pool2d(encoded, output_size=1).flatten(1)
        return pooled.view(batch_size, frames, self.feature_dim).mean(dim=1)


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
        backbone: str = "resnet18",
        pretrained_backbone: bool = True,
        projection_hidden_dim: int | None = None,
        projection_dim: int = 128,
    ) -> None:
        super().__init__()
        self.encoder = TemporalResNetEncoder(backbone=backbone, pretrained_backbone=pretrained_backbone)
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
        state_dict = state["model"] if isinstance(state, dict) and "model" in state else state
        model.load_state_dict(state_dict, strict=False)
    return model


def build_supcon_model(config: dict) -> SupConPretrainingModel:
    model_config = config["model"]
    model = SupConPretrainingModel(
        backbone=model_config.get("backbone", "resnet18"),
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
