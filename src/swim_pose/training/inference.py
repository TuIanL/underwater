from __future__ import annotations

from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from ..constants import KEYPOINT_NAMES
from ..io import write_jsonl
from .config import load_config
from .dataset import PoseDataset, UnlabeledFrameDataset
from .model import build_model


def run_inference(
    config_path: str | Path,
    checkpoint_path: str | Path,
    index_path: str | Path,
    output_path: str | Path,
    labeled: bool = True,
) -> Path:
    config = load_config(config_path)
    input_size = (config["dataset"]["input_width"], config["dataset"]["input_height"])
    heatmap_size = (config["dataset"]["heatmap_width"], config["dataset"]["heatmap_height"])
    image_root = config["dataset"]["image_root"]
    if labeled:
        dataset = PoseDataset(index_path, image_root=image_root, input_size=input_size, heatmap_size=heatmap_size)
    else:
        dataset = UnlabeledFrameDataset(index_path, image_root=image_root, input_size=input_size)

    loader = DataLoader(dataset, batch_size=1, shuffle=False)
    model = build_model(config, len(KEYPOINT_NAMES))
    state = torch.load(checkpoint_path, map_location="cpu")
    model.load_state_dict(state["model"] if isinstance(state, dict) and "model" in state else state, strict=False)
    model.eval()

    outputs: list[dict] = []
    with torch.no_grad():
        for batch in loader:
            predictions = model(batch["image"])
            outputs.extend(_decode_batch(batch, predictions))
    write_jsonl(output_path, outputs)
    return Path(output_path)


def _decode_batch(batch: dict, predictions: dict[str, torch.Tensor]) -> list[dict]:
    heatmaps = predictions["heatmaps"].sigmoid()
    visibility_probs = F.softmax(predictions["visibility_logits"], dim=-1)
    batch_rows: list[dict] = []
    for batch_index in range(heatmaps.shape[0]):
        original_width = float(_unwrap(batch.get("original_width"), batch_index))
        original_height = float(_unwrap(batch.get("original_height"), batch_index))
        row = {
            "annotation_path": _unwrap(batch.get("annotation_path"), batch_index),
            "clip_id": _unwrap(batch.get("clip_id"), batch_index),
            "athlete_id": _unwrap(batch.get("athlete_id"), batch_index),
            "session_id": _unwrap(batch.get("session_id"), batch_index),
            "frame_index": int(_unwrap(batch.get("frame_index"), batch_index)),
            "source_view": _unwrap(batch.get("source_view"), batch_index),
            "image_path": _unwrap(batch.get("image_path"), batch_index),
            "points": {},
        }
        for keypoint_index, name in enumerate(KEYPOINT_NAMES):
            heatmap = heatmaps[batch_index, keypoint_index]
            flat_index = int(torch.argmax(heatmap).item())
            y_index = flat_index // heatmap.shape[1]
            x_index = flat_index % heatmap.shape[1]
            confidence = float(torch.max(heatmap).cpu())
            visibility = int(torch.argmax(visibility_probs[batch_index, keypoint_index]).cpu())
            row["points"][name] = {
                "x": float((x_index + 0.5) / heatmap.shape[1] * original_width),
                "y": float((y_index + 0.5) / heatmap.shape[0] * original_height),
                "confidence": confidence,
                "visibility": visibility,
            }
        batch_rows.append(row)
    return batch_rows


def _unwrap(value: object, index: int) -> object:
    if value is None:
        return ""
    if isinstance(value, list):
        return value[index]
    if hasattr(value, "ndim"):
        scalar = value[index]
        if hasattr(scalar, "item"):
            return scalar.item()
        return scalar
    if hasattr(value, "__getitem__") and not isinstance(value, (str, bytes, dict)):
        try:
            return value[index]
        except Exception:
            return value
    return value
