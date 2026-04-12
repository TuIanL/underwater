from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image
from torch.utils.data import Dataset

from ..annotations import validate_annotation
from ..constants import KEYPOINT_NAMES
from ..io import read_csv_rows, read_json


class PoseDataset(Dataset):
    def __init__(
        self,
        index_path: str | Path,
        image_root: str | Path,
        input_size: tuple[int, int],
        heatmap_size: tuple[int, int],
    ) -> None:
        self.rows = read_csv_rows(index_path)
        self.image_root = Path(image_root)
        self.input_size = input_size
        self.heatmap_size = heatmap_size

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> dict:
        import torch

        row = self.rows[index]
        annotation = read_json(row["annotation_path"])
        errors = validate_annotation(annotation)
        if errors:
            raise ValueError(f"Invalid annotation {row['annotation_path']}: {'; '.join(errors)}")

        resolved_image_path = self._resolve_image_path(annotation.get("image_path", ""))
        image, original_size = _load_image(resolved_image_path, self.input_size)
        heatmaps, visibility, points = build_targets(
            annotation=annotation,
            original_size=original_size,
            input_size=self.input_size,
            heatmap_size=self.heatmap_size,
        )
        return {
            "image": image,
            "heatmaps": torch.from_numpy(heatmaps),
            "visibility": torch.from_numpy(visibility),
            "points": torch.from_numpy(points),
            "annotation_path": row["annotation_path"],
            "clip_id": row.get("clip_id", ""),
            "athlete_id": row.get("athlete_id", ""),
            "session_id": row.get("session_id", ""),
            "frame_index": _safe_int(row.get("frame_index", 0)),
            "source_view": row.get("source_view", ""),
            "image_path": annotation.get("image_path", ""),
            "original_width": original_size[0],
            "original_height": original_size[1],
        }

    def _resolve_image_path(self, image_path: str) -> Path:
        candidate = Path(image_path)
        if candidate.is_absolute():
            return candidate
        if candidate.exists():
            return candidate
        return self.image_root / image_path


class UnlabeledFrameDataset(Dataset):
    def __init__(self, index_path: str | Path, image_root: str | Path, input_size: tuple[int, int]) -> None:
        self.rows = read_csv_rows(index_path)
        self.image_root = Path(image_root)
        self.input_size = input_size

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> dict:
        row = self.rows[index]
        image_path = row.get("image_path", "")
        image, original_size = _load_image(self._resolve_image_path(image_path), self.input_size)
        return {
            "image": image,
            "image_path": image_path,
            "clip_id": row.get("clip_id", ""),
            "athlete_id": row.get("athlete_id", ""),
            "session_id": row.get("session_id", ""),
            "frame_index": _safe_int(row.get("frame_index", 0)),
            "source_view": row.get("source_view", ""),
            "original_width": original_size[0],
            "original_height": original_size[1],
        }

    def _resolve_image_path(self, image_path: str) -> Path:
        candidate = Path(image_path)
        if candidate.is_absolute():
            return candidate
        if candidate.exists():
            return candidate
        return self.image_root / image_path


class TemporalUnlabeledFrameDataset(UnlabeledFrameDataset):
    def __init__(self, index_path: str | Path, image_root: str | Path, input_size: tuple[int, int]) -> None:
        super().__init__(index_path=index_path, image_root=image_root, input_size=input_size)
        self.temporal_pairs = self._build_temporal_pairs()

    def __getitem__(self, index: int) -> dict:
        current = super().__getitem__(index)
        pair_index = self.temporal_pairs[index]
        if pair_index is None:
            current["temporal_image"] = current["image"]
            current["has_temporal_pair"] = False
            return current

        neighbor_row = self.rows[pair_index]
        temporal_image, _ = _load_image(self._resolve_image_path(neighbor_row.get("image_path", "")), self.input_size)
        current["temporal_image"] = temporal_image
        current["has_temporal_pair"] = True
        return current

    def _build_temporal_pairs(self) -> list[int | None]:
        ordered = sorted(
            enumerate(self.rows),
            key=lambda item: (
                item[1].get("clip_id", ""),
                item[1].get("source_view", ""),
                _safe_int(item[1].get("frame_index", 0)),
            ),
        )
        temporal_pairs: list[int | None] = [None] * len(self.rows)
        for current, nxt in zip(ordered, ordered[1:]):
            current_index, current_row = current
            next_index, next_row = nxt
            same_stream = (
                current_row.get("clip_id", "") == next_row.get("clip_id", "")
                and current_row.get("source_view", "") == next_row.get("source_view", "")
            )
            if same_stream:
                temporal_pairs[current_index] = next_index
        return temporal_pairs


def build_targets(
    annotation: dict,
    original_size: tuple[int, int],
    input_size: tuple[int, int],
    heatmap_size: tuple[int, int],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    input_width, input_height = input_size
    heatmap_width, heatmap_height = heatmap_size
    original_width, original_height = original_size
    points = np.zeros((len(KEYPOINT_NAMES), 2), dtype=np.float32)
    heatmaps = np.zeros((len(KEYPOINT_NAMES), heatmap_height, heatmap_width), dtype=np.float32)
    visibility = np.zeros((len(KEYPOINT_NAMES),), dtype=np.int64)
    for index, keypoint in enumerate(KEYPOINT_NAMES):
        point = annotation["points"][keypoint]
        visibility[index] = point["visibility"]
        if point["visibility"] == 0 or point["x"] is None or point["y"] is None:
            continue
        scaled_x = float(point["x"]) / original_width * input_width
        scaled_y = float(point["y"]) / original_height * input_height
        points[index] = np.array([scaled_x, scaled_y], dtype=np.float32)
        heatmaps[index] = gaussian_heatmap(
            x=scaled_x / input_width * heatmap_width,
            y=scaled_y / input_height * heatmap_height,
            width=heatmap_width,
            height=heatmap_height,
        )
    return heatmaps, visibility, points


def _load_image(path: Path, input_size: tuple[int, int]):
    import torch

    image = Image.open(path).convert("RGB")
    original_size = image.size
    image = image.resize(input_size, Image.Resampling.BILINEAR)
    array = np.asarray(image, dtype=np.float32) / 255.0
    tensor = torch.from_numpy(array).permute(2, 0, 1)
    return tensor, original_size


def gaussian_heatmap(x: float, y: float, width: int, height: int, sigma: float = 1.8) -> np.ndarray:
    xs = np.arange(width, dtype=np.float32)
    ys = np.arange(height, dtype=np.float32)
    grid_x, grid_y = np.meshgrid(xs, ys)
    exponent = ((grid_x - x) ** 2 + (grid_y - y) ** 2) / (2 * sigma**2)
    return np.exp(-exponent)


def _resolve_existing_path(image_path: str) -> Path:
    path = Path(image_path)
    if not path.exists():
        raise FileNotFoundError(f"Image path not found: {image_path}")
    return path


def _safe_int(value: object) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return 0
