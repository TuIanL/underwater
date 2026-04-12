from __future__ import annotations

from pathlib import Path

import torch
from torch import nn
from torchvision.models import ResNet18_Weights, ResNet34_Weights, resnet18, resnet34


class ResNetHeatmapModel(nn.Module):
    def __init__(self, num_keypoints: int, backbone: str = "resnet18", pretrained_backbone: bool = True) -> None:
        super().__init__()
        if backbone == "resnet34":
            weights = ResNet34_Weights.DEFAULT if pretrained_backbone else None
            backbone_model = resnet34(weights=weights)
        else:
            weights = ResNet18_Weights.DEFAULT if pretrained_backbone else None
            backbone_model = resnet18(weights=weights)

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
            nn.Linear(512, num_keypoints * 3),
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

