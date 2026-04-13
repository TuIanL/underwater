from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn


def supervised_pose_loss(
    predictions: dict[str, torch.Tensor],
    target_heatmaps: torch.Tensor,
    target_visibility: torch.Tensor,
    visibility_loss_weight: float,
) -> tuple[torch.Tensor, dict[str, float]]:
    predicted_heatmaps = predictions["heatmaps"]
    visibility_logits = predictions["visibility_logits"]

    point_mask = (target_visibility > 0).float().unsqueeze(-1).unsqueeze(-1)
    heatmap_loss = F.mse_loss(predicted_heatmaps, target_heatmaps, reduction="none")
    heatmap_loss = (heatmap_loss * point_mask).mean()
    visibility_loss = F.cross_entropy(
        visibility_logits.reshape(-1, 3),
        target_visibility.reshape(-1),
    )
    total = heatmap_loss + visibility_loss_weight * visibility_loss
    return total, {
        "heatmap_loss": float(heatmap_loss.detach().cpu()),
        "visibility_loss": float(visibility_loss.detach().cpu()),
    }


class SupConLoss(nn.Module):
    def __init__(self, temperature: float = 0.07) -> None:
        super().__init__()
        self.temperature = temperature

    def forward(self, projections: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        if projections.ndim != 2:
            raise ValueError(f"Expected 2D projections, got shape {tuple(projections.shape)}")
        if labels.ndim != 1:
            labels = labels.reshape(-1)
        if projections.shape[0] != labels.shape[0]:
            raise ValueError("Number of projections and labels must match")

        logits = torch.matmul(projections, projections.T) / self.temperature
        logits = logits - logits.max(dim=1, keepdim=True).values.detach()

        self_mask = torch.eye(logits.shape[0], device=logits.device, dtype=torch.bool)
        positive_mask = labels.unsqueeze(0).eq(labels.unsqueeze(1)) & ~self_mask
        if torch.any(positive_mask.sum(dim=1) == 0):
            raise ValueError("SupConLoss requires at least one positive pair for every anchor")

        log_denominator = torch.logsumexp(logits.masked_fill(self_mask, float("-inf")), dim=1, keepdim=True)
        log_prob = logits - log_denominator
        mean_log_prob = (positive_mask.float() * log_prob).sum(dim=1) / positive_mask.sum(dim=1).clamp_min(1)
        return -mean_log_prob.mean()


def consistency_loss(
    weak_predictions: dict[str, torch.Tensor],
    strong_predictions: dict[str, torch.Tensor],
    threshold: float,
) -> torch.Tensor:
    weak_heatmaps = weak_predictions["heatmaps"].detach().sigmoid()
    strong_heatmaps = strong_predictions["heatmaps"].sigmoid()
    confidence_mask = (weak_heatmaps.amax(dim=(-1, -2), keepdim=True) >= threshold).float()
    return ((weak_heatmaps - strong_heatmaps) ** 2 * confidence_mask).mean()


def temporal_smoothness_loss(
    current_predictions: dict[str, torch.Tensor],
    temporal_predictions: dict[str, torch.Tensor],
    has_temporal_pair: torch.Tensor,
    threshold: float,
) -> torch.Tensor:
    current_coords, current_confidence = _decode_heatmap_coordinates(current_predictions["heatmaps"])
    temporal_coords, temporal_confidence = _decode_heatmap_coordinates(temporal_predictions["heatmaps"])
    pair_mask = (current_confidence >= threshold) & (temporal_confidence >= threshold)
    pair_mask = pair_mask & has_temporal_pair.reshape(-1, 1).bool()
    if not torch.any(pair_mask):
        return current_predictions["heatmaps"].new_tensor(0.0)
    squared_distance = ((current_coords - temporal_coords) ** 2).sum(dim=-1)
    return squared_distance[pair_mask].mean()


def _decode_heatmap_coordinates(heatmaps: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    probabilities = heatmaps.sigmoid()
    batch_size, num_keypoints, height, width = probabilities.shape
    flat = probabilities.reshape(batch_size, num_keypoints, -1)
    flat_indices = flat.argmax(dim=-1)
    confidence = flat.amax(dim=-1)
    y = (flat_indices // width).float() / max(height - 1, 1)
    x = (flat_indices % width).float() / max(width - 1, 1)
    coordinates = torch.stack((x, y), dim=-1)
    return coordinates, confidence
