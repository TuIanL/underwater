from __future__ import annotations

from collections import defaultdict
from typing import Iterable

from ..constants import KEYPOINT_NAMES


def postprocess_enabled(config: dict) -> bool:
    return bool(config.get("postprocess", {}).get("enabled", False))


def apply_temporal_postprocessing(rows: list[dict], config: dict) -> list[dict]:
    if not rows or not postprocess_enabled(config):
        return rows

    postprocess_config = config.get("postprocess", {})
    method = str(postprocess_config.get("method", "ema")).strip().lower()
    if method != "ema":
        raise ValueError(f"Unsupported temporal postprocessing method '{method}'. Expected 'ema'.")

    max_alpha = min(max(float(postprocess_config.get("alpha", 0.65)), 0.0), 1.0)
    min_alpha = min(max(float(postprocess_config.get("min_alpha", min(max_alpha, 0.2))), 0.0), max_alpha)
    confidence_floor = min(max(float(postprocess_config.get("confidence_floor", 0.1)), 0.0), 1.0)

    groups: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for row in rows:
        groups[(str(row.get("clip_id", "")), str(row.get("source_view", "")))].append(row)

    for group_rows in groups.values():
        ordered = sorted(group_rows, key=lambda item: int(item.get("frame_index", 0)))
        previous_filtered: dict[str, dict | None] = {name: None for name in KEYPOINT_NAMES}
        previous_frame_index: int | None = None
        for row in ordered:
            current_frame = int(row.get("frame_index", 0))
            if previous_frame_index is None or current_frame - previous_frame_index > 1:
                previous_filtered = {name: None for name in KEYPOINT_NAMES}

            filtered_points = {}
            for name in KEYPOINT_NAMES:
                raw_point = row.get("points", {}).get(name, {})
                filtered_point = _filter_point(
                    raw_point=raw_point,
                    previous_filtered=previous_filtered.get(name),
                    min_alpha=min_alpha,
                    max_alpha=max_alpha,
                    confidence_floor=confidence_floor,
                )
                filtered_points[name] = filtered_point
                previous_filtered[name] = filtered_point
            row["filtered_points"] = filtered_points
            previous_frame_index = current_frame
    return rows


def build_filtered_variant_rows(rows: Iterable[dict]) -> list[dict]:
    filtered_rows: list[dict] = []
    for row in rows:
        if "filtered_points" not in row:
            continue
        filtered_row = {key: value for key, value in row.items() if key != "points"}
        filtered_row["points"] = row["filtered_points"]
        filtered_row["prediction_variant"] = "filtered"
        filtered_rows.append(filtered_row)
    return filtered_rows


def _filter_point(
    raw_point: dict,
    previous_filtered: dict | None,
    min_alpha: float,
    max_alpha: float,
    confidence_floor: float,
) -> dict:
    x = raw_point.get("x")
    y = raw_point.get("y")
    confidence = float(raw_point.get("confidence", 0.0))
    visibility = int(raw_point.get("visibility", 0))

    if x is None or y is None or visibility == 0:
        return {
            "x": None,
            "y": None,
            "confidence": confidence,
            "visibility": visibility,
        }

    if previous_filtered is None or previous_filtered.get("x") is None or previous_filtered.get("y") is None:
        return {
            "x": float(x),
            "y": float(y),
            "confidence": confidence,
            "visibility": visibility,
        }

    confidence_weight = 0.0 if confidence <= confidence_floor else min(
        (confidence - confidence_floor) / max(1.0 - confidence_floor, 1e-6),
        1.0,
    )
    alpha = min_alpha + (max_alpha - min_alpha) * confidence_weight
    previous_x = float(previous_filtered["x"])
    previous_y = float(previous_filtered["y"])

    return {
        "x": alpha * float(x) + (1.0 - alpha) * previous_x,
        "y": alpha * float(y) + (1.0 - alpha) * previous_y,
        "confidence": confidence,
        "visibility": visibility,
    }
