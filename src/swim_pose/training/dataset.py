from __future__ import annotations

from bisect import bisect_left
import random
from collections import defaultdict
from pathlib import Path

import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset

from ..annotations import validate_annotation
from ..constants import KEYPOINT_NAMES
from ..io import read_csv_rows, read_json
from ..pathing import find_repo_root, resolve_persisted_source_path, resolve_repo_managed_path


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
        self.repo_root = find_repo_root()

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> dict:
        import torch

        row = self.rows[index]
        annotation = read_json(resolve_repo_managed_path(row["annotation_path"], self.repo_root))
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


class TemporalPoseDataset(PoseDataset):
    def __init__(
        self,
        index_path: str | Path,
        image_root: str | Path,
        input_size: tuple[int, int],
        heatmap_size: tuple[int, int],
        bridge_input_size: tuple[int, int],
        bridge_clip_length: int,
        bridge_frame_stride: int = 1,
    ) -> None:
        super().__init__(
            index_path=index_path,
            image_root=image_root,
            input_size=input_size,
            heatmap_size=heatmap_size,
        )
        self.bridge_input_size = bridge_input_size
        self.bridge_clip_length = max(int(bridge_clip_length), 1)
        self.bridge_frame_stride = max(int(bridge_frame_stride), 1)
        self.temporal_groups = self._build_temporal_groups()

    def __getitem__(self, index: int) -> dict:
        sample = super().__getitem__(index)
        sample["bridge_clip"], sample["bridge_frame_indices"] = self._build_bridge_clip(index)
        return sample

    def _build_temporal_groups(self) -> dict[tuple[str, str], list[tuple[int, int]]]:
        groups: dict[tuple[str, str], list[tuple[int, int]]] = defaultdict(list)
        for row_index, row in enumerate(self.rows):
            key = (row.get("clip_id", ""), row.get("source_view", ""))
            groups[key].append((_safe_int(row.get("frame_index", 0)), row_index))
        for key in groups:
            groups[key].sort(key=lambda item: item[0])
        return groups

    def _build_bridge_clip(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        row = self.rows[index]
        key = (row.get("clip_id", ""), row.get("source_view", ""))
        group = self.temporal_groups.get(key)
        if not group:
            image = self._load_row_image(row, self.bridge_input_size)
            frames = [image.clone() for _ in range(self.bridge_clip_length)]
            frame_indices = [_safe_int(row.get("frame_index", 0)) for _ in range(self.bridge_clip_length)]
            return torch.stack(frames, dim=1), torch.tensor(frame_indices, dtype=torch.long)

        current_frame = _safe_int(row.get("frame_index", 0))
        offsets = _centered_temporal_offsets(self.bridge_clip_length, self.bridge_frame_stride)
        selected_rows: list[dict[str, str]] = []
        selected_frame_indices: list[int] = []
        for offset in offsets:
            matched_frame_index, matched_row_index = _nearest_group_row(group, current_frame + offset)
            selected_rows.append(self.rows[matched_row_index])
            selected_frame_indices.append(matched_frame_index)
        frames = [self._load_row_image(selected_row, self.bridge_input_size) for selected_row in selected_rows]
        return torch.stack(frames, dim=1), torch.tensor(selected_frame_indices, dtype=torch.long)

    def _load_row_image(self, row: dict[str, str], input_size: tuple[int, int]) -> torch.Tensor:
        image_path = row.get("image_path", "")
        if not image_path:
            annotation = read_json(resolve_repo_managed_path(row["annotation_path"], self.repo_root))
            image_path = annotation.get("image_path", "")
        image, _ = _load_image(self._resolve_image_path(image_path), input_size)
        return image


class SupConVideoDataset(Dataset):
    def __init__(
        self,
        index_path: str | Path,
        input_size: tuple[int, int],
        clip_length: int,
        frame_stride: int = 2,
        temporal_jitter: int = 1,
        crop_scale_range: tuple[float, float] = (0.7, 1.0),
        color_jitter_strength: float = 0.4,
        grayscale_prob: float = 0.2,
        blur_prob: float = 0.5,
        blur_kernel_size: int = 5,
        valid_only: bool = True,
    ) -> None:
        rows = read_csv_rows(index_path)
        if valid_only:
            rows = [row for row in rows if row.get("validation_status", "valid") == "valid"]
        if not rows:
            raise ValueError(f"No valid Phase 1 videos found in index: {index_path}")
        self.rows = rows
        self.input_size = input_size
        self.clip_length = max(int(clip_length), 1)
        self.frame_stride = max(int(frame_stride), 1)
        self.temporal_jitter = max(int(temporal_jitter), 0)
        self.crop_scale_range = crop_scale_range
        self.color_jitter_strength = max(float(color_jitter_strength), 0.0)
        self.grayscale_prob = min(max(float(grayscale_prob), 0.0), 1.0)
        self.blur_prob = min(max(float(blur_prob), 0.0), 1.0)
        self.blur_kernel_size = max(int(blur_kernel_size), 1)
        if self.blur_kernel_size % 2 == 0:
            self.blur_kernel_size += 1
        self.repo_root = find_repo_root()
        labels = sorted({row["stroke_label"] for row in self.rows})
        self.label_to_index = {label: index for index, label in enumerate(labels)}

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> dict:
        row = self.rows[index]
        label = self.label_to_index[row["stroke_label"]]
        frame_count = max(_safe_int(row.get("frame_count", 0)), 1)
        resolved_path = self._resolve_video_path(row["video_path"])
        clip_raw, frame_indices = self._sample_raw_clip(resolved_path, frame_count)
        view_1 = self._augment_clip(clip_raw)
        view_2 = self._augment_clip(clip_raw)
        return {
            "view_1": view_1,
            "view_2": view_2,
            "label": torch.tensor(label, dtype=torch.long),
            "clip_frame_indices": frame_indices,
            "stroke_label": row["stroke_label"],
            "video_id": row.get("video_id", ""),
            "video_path": row["video_path"],
            "athlete_id": row.get("athlete_id", ""),
            "session_id": row.get("session_id", ""),
            "take_id": row.get("take_id", ""),
        }

    def _sample_raw_clip(self, path: Path, frame_count: int) -> tuple[torch.Tensor, torch.Tensor]:
        frame_indices = _sample_temporal_indices(
            frame_count=frame_count,
            clip_length=self.clip_length,
            frame_stride=self.frame_stride,
            temporal_jitter=self.temporal_jitter,
        )
        clip = _load_video_clip(path, frame_indices, self.input_size)
        return clip, torch.tensor(frame_indices, dtype=torch.long)

    def _augment_clip(self, clip: torch.Tensor) -> torch.Tensor:
        return _augment_video_clip(
            clip=clip.clone(),
            crop_scale_range=self.crop_scale_range,
            color_jitter_strength=self.color_jitter_strength,
            grayscale_prob=self.grayscale_prob,
            blur_prob=self.blur_prob,
            blur_kernel_size=self.blur_kernel_size,
        )

    def _resolve_video_path(self, video_path: str) -> Path:
        return resolve_persisted_source_path(video_path, self.repo_root)


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
    image = Image.open(path).convert("RGB")
    original_size = image.size
    image = image.resize(input_size, Image.Resampling.BILINEAR)
    array = np.asarray(image, dtype=np.float32) / 255.0
    tensor = torch.from_numpy(array).permute(2, 0, 1)
    return tensor, original_size


def _load_video_clip(path: Path, frame_indices: list[int], input_size: tuple[int, int]) -> torch.Tensor:
    try:
        import cv2
    except ImportError as exc:
        raise RuntimeError("opencv-python is required for video clip loading.") from exc

    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        raise RuntimeError(f"Could not open video file: {path}")

    frames: list[torch.Tensor] = []
    previous_frame: np.ndarray | None = None
    width, height = input_size
    try:
        for frame_index in frame_indices:
            capture.set(cv2.CAP_PROP_POS_FRAMES, int(frame_index))
            success, frame = capture.read()
            if not success:
                if previous_frame is None:
                    raise RuntimeError(f"Could not decode frame {frame_index} from {path}")
                frame = previous_frame.copy()
            else:
                previous_frame = frame
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            if frame.shape[1] != width or frame.shape[0] != height:
                frame = cv2.resize(frame, (width, height), interpolation=cv2.INTER_LINEAR)
            tensor = torch.from_numpy(frame).float().permute(2, 0, 1) / 255.0
            frames.append(tensor)
    finally:
        capture.release()

    return torch.stack(frames, dim=1)


def _sample_temporal_indices(
    frame_count: int,
    clip_length: int,
    frame_stride: int,
    temporal_jitter: int,
) -> list[int]:
    effective_stride = max(1, frame_stride + random.randint(-temporal_jitter, temporal_jitter))
    temporal_span = effective_stride * max(clip_length - 1, 0) + 1
    max_start = max(frame_count - temporal_span, 0)
    start_index = random.randint(0, max_start) if max_start > 0 else 0
    return [min(start_index + step * effective_stride, frame_count - 1) for step in range(clip_length)]


def _augment_video_clip(
    clip: torch.Tensor,
    crop_scale_range: tuple[float, float],
    color_jitter_strength: float,
    grayscale_prob: float,
    blur_prob: float,
    blur_kernel_size: int,
) -> torch.Tensor:
    augmented = _random_resized_crop_clip(clip, crop_scale_range)
    augmented = _apply_color_jitter(augmented, color_jitter_strength)
    if random.random() < grayscale_prob:
        grayscale = augmented.mean(dim=0, keepdim=True)
        augmented = grayscale.repeat(augmented.shape[0], 1, 1, 1)
    if random.random() < blur_prob:
        sigma = random.uniform(0.5, 1.5)
        augmented = _apply_gaussian_blur(augmented, kernel_size=blur_kernel_size, sigma=sigma)
    return torch.clamp(augmented, 0.0, 1.0)


def _random_resized_crop_clip(clip: torch.Tensor, crop_scale_range: tuple[float, float]) -> torch.Tensor:
    _, _, height, width = clip.shape
    min_scale, max_scale = crop_scale_range
    scale = random.uniform(min_scale, max_scale)
    crop_height = max(1, int(height * scale))
    crop_width = max(1, int(width * scale))
    top = random.randint(0, max(height - crop_height, 0))
    left = random.randint(0, max(width - crop_width, 0))
    cropped = clip[:, :, top:top + crop_height, left:left + crop_width]
    resized = F.interpolate(
        cropped.permute(1, 0, 2, 3),
        size=(height, width),
        mode="bilinear",
        align_corners=False,
    )
    return resized.permute(1, 0, 2, 3)


def _apply_color_jitter(clip: torch.Tensor, strength: float) -> torch.Tensor:
    if strength <= 0:
        return clip
    brightness = 1.0 + random.uniform(-strength, strength)
    contrast = 1.0 + random.uniform(-strength, strength)
    saturation = 1.0 + random.uniform(-strength, strength)

    jittered = clip * brightness
    frame_mean = jittered.mean(dim=(0, 2, 3), keepdim=True)
    jittered = (jittered - frame_mean) * contrast + frame_mean

    grayscale = jittered.mean(dim=0, keepdim=True)
    jittered = (jittered - grayscale) * saturation + grayscale
    return jittered


def _apply_gaussian_blur(clip: torch.Tensor, kernel_size: int, sigma: float) -> torch.Tensor:
    radius = kernel_size // 2
    coords = torch.arange(-radius, radius + 1, dtype=clip.dtype)
    kernel_1d = torch.exp(-(coords**2) / max(2 * sigma**2, 1e-6))
    kernel_1d = kernel_1d / kernel_1d.sum()
    kernel_2d = torch.outer(kernel_1d, kernel_1d)
    kernel = kernel_2d.expand(clip.shape[0], 1, kernel_size, kernel_size).contiguous()
    frames = clip.permute(1, 0, 2, 3)
    blurred = F.conv2d(frames, kernel, padding=radius, groups=clip.shape[0])
    return blurred.permute(1, 0, 2, 3)


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


def _centered_temporal_offsets(clip_length: int, frame_stride: int) -> list[int]:
    anchor = clip_length // 2
    return [(index - anchor) * frame_stride for index in range(clip_length)]


def _nearest_group_row(group: list[tuple[int, int]], desired_frame_index: int) -> tuple[int, int]:
    frame_indices = [frame_index for frame_index, _ in group]
    position = bisect_left(frame_indices, desired_frame_index)
    if position <= 0:
        return group[0]
    if position >= len(group):
        return group[-1]
    previous = group[position - 1]
    current = group[position]
    if abs(previous[0] - desired_frame_index) <= abs(current[0] - desired_frame_index):
        return previous
    return current
