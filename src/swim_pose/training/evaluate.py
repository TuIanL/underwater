from __future__ import annotations

import math
from collections import defaultdict
from pathlib import Path

from ..constants import KEYPOINT_NAMES
from ..io import read_csv_rows, read_json, read_jsonl, write_json
from ..pathing import find_repo_root, resolve_repo_managed_path


def evaluate_predictions_file(predictions_path: str | Path, annotations_path: str | Path, output_path: str | Path) -> Path:
    predictions = read_jsonl(predictions_path)
    annotation_lookup = load_annotation_lookup(annotations_path)

    raw_report = _evaluate_variant(predictions, annotation_lookup, point_field="points")
    report = {
        "overall": raw_report["overall"],
        "per_joint": raw_report["per_joint"],
        "temporal_stability": {"raw": raw_report["temporal_stability"]},
    }

    if any("filtered_points" in row for row in predictions):
        filtered_report = _evaluate_variant(predictions, annotation_lookup, point_field="filtered_points")
        report["filtered_overall"] = filtered_report["overall"]
        report["filtered_per_joint"] = filtered_report["per_joint"]
        report["temporal_stability"]["filtered"] = filtered_report["temporal_stability"]

    write_json(output_path, report)
    return Path(output_path)


def _evaluate_variant(predictions: list[dict], annotation_lookup: dict[str, dict], point_field: str) -> dict[str, object]:
    aggregate_errors: list[float] = []
    visible_errors: list[float] = []
    occluded_errors: list[float] = []
    per_joint = defaultdict(list)
    pck_005 = []
    pck_010 = []
    temporal_groups = defaultdict(list)

    for prediction in predictions:
        points = prediction.get(point_field)
        if not isinstance(points, dict):
            continue
        annotation = resolve_annotation(annotation_lookup, prediction)
        if annotation is None:
            continue
        normalizer = annotation_scale(annotation)
        temporal_groups[(prediction["clip_id"], prediction["source_view"])].append(
            {
                "frame_index": int(prediction["frame_index"]),
                "points": points,
            }
        )
        for keypoint in KEYPOINT_NAMES:
            pred_point = points[keypoint]
            gt_point = annotation["points"][keypoint]
            if gt_point["visibility"] == 0:
                continue
            if pred_point.get("x") is None or pred_point.get("y") is None:
                continue
            error = math.dist((pred_point["x"], pred_point["y"]), (gt_point["x"], gt_point["y"])) / max(normalizer, 1e-6)
            aggregate_errors.append(error)
            per_joint[keypoint].append(error)
            if gt_point["visibility"] == 2:
                visible_errors.append(error)
            else:
                occluded_errors.append(error)
            pck_005.append(1.0 if error <= 0.05 else 0.0)
            pck_010.append(1.0 if error <= 0.10 else 0.0)

    temporal_stability = summarize_temporal_stability(temporal_groups)
    return {
        "overall": {
            "mean_normalized_error": mean_or_zero(aggregate_errors),
            "pck@0.05": mean_or_zero(pck_005),
            "pck@0.10": mean_or_zero(pck_010),
            "visible_mean_error": mean_or_zero(visible_errors),
            "occluded_mean_error": mean_or_zero(occluded_errors),
            "temporal_jitter": temporal_stability["mean_midpoint_residual"],
        },
        "per_joint": {
            keypoint: {
                "mean_normalized_error": mean_or_zero(errors),
                "count": len(errors),
            }
            for keypoint, errors in per_joint.items()
        },
        "temporal_stability": temporal_stability,
    }


def summarize_temporal_stability(groups: dict[tuple[str, str], list[dict]]) -> dict[str, object]:
    frame_displacements: list[float] = []
    midpoint_residuals: list[float] = []
    confidence_weighted_residuals: list[float] = []
    top_unstable_windows: list[dict[str, object]] = []

    for (clip_id, source_view), group_predictions in groups.items():
        ordered = sorted(group_predictions, key=lambda row: int(row["frame_index"]))
        for earlier, later in zip(ordered, ordered[1:]):
            displacement_values = _pointwise_distances(earlier["points"], later["points"])
            frame_displacements.extend(displacement_values)
        for earlier, current, later in zip(ordered, ordered[1:], ordered[2:]):
            residual_values = _midpoint_residuals(earlier["points"], current["points"], later["points"])
            if residual_values:
                window_mean = sum(residual_values) / len(residual_values)
                midpoint_residuals.extend(residual_values)
                confidence_weighted_residuals.append(window_mean * _confidence_weight(current["points"]))
                top_unstable_windows.append(
                    {
                        "clip_id": clip_id,
                        "source_view": source_view,
                        "frame_index": int(current["frame_index"]),
                        "mean_midpoint_residual": window_mean,
                    }
                )

    ranked_windows = sorted(top_unstable_windows, key=lambda row: row["mean_midpoint_residual"], reverse=True)[:10]
    return {
        "mean_frame_displacement": mean_or_zero(frame_displacements),
        "p95_frame_displacement": percentile(frame_displacements, 0.95),
        "mean_midpoint_residual": mean_or_zero(midpoint_residuals),
        "p95_midpoint_residual": percentile(midpoint_residuals, 0.95),
        "confidence_weighted_midpoint_residual": mean_or_zero(confidence_weighted_residuals),
        "top_unstable_windows": ranked_windows,
    }


def load_annotation_lookup(path: str | Path) -> dict[str, dict]:
    resolved = Path(path)
    repo_root = find_repo_root()
    if resolved.suffix == ".csv":
        rows = read_csv_rows(resolved)
        lookup = {}
        for row in rows:
            annotation = read_json(resolve_repo_managed_path(row["annotation_path"], repo_root))
            lookup[row["annotation_path"]] = annotation
            lookup[_annotation_key(annotation)] = annotation
        return lookup
    if resolved.suffix == ".json":
        annotation = read_json(resolved)
        return {resolved.as_posix(): annotation, _annotation_key(annotation): annotation}
    raise ValueError("Annotations must be a CSV annotation index or a JSON annotation file.")


def resolve_annotation(lookup: dict[str, dict], prediction: dict) -> dict | None:
    if prediction.get("annotation_path") in lookup:
        return lookup[prediction["annotation_path"]]
    return lookup.get(_prediction_key(prediction))


def annotation_scale(annotation: dict) -> float:
    xs = []
    ys = []
    for point in annotation["points"].values():
        if point["visibility"] > 0 and point["x"] is not None and point["y"] is not None:
            xs.append(float(point["x"]))
            ys.append(float(point["y"]))
    if len(xs) < 2 or len(ys) < 2:
        return 1.0
    width = max(xs) - min(xs)
    height = max(ys) - min(ys)
    return math.sqrt(width**2 + height**2)


def mean_or_zero(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def percentile(values: list[float], quantile: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = min(max(int(round((len(ordered) - 1) * quantile)), 0), len(ordered) - 1)
    return ordered[index]


def _pointwise_distances(first_points: dict, second_points: dict) -> list[float]:
    distances: list[float] = []
    for keypoint in KEYPOINT_NAMES:
        first = first_points.get(keypoint, {})
        second = second_points.get(keypoint, {})
        if first.get("x") is None or first.get("y") is None or second.get("x") is None or second.get("y") is None:
            continue
        distances.append(math.dist((first["x"], first["y"]), (second["x"], second["y"])))
    return distances


def _midpoint_residuals(previous_points: dict, current_points: dict, next_points: dict) -> list[float]:
    residuals: list[float] = []
    for keypoint in KEYPOINT_NAMES:
        previous = previous_points.get(keypoint, {})
        current = current_points.get(keypoint, {})
        later = next_points.get(keypoint, {})
        if (
            previous.get("x") is None
            or previous.get("y") is None
            or current.get("x") is None
            or current.get("y") is None
            or later.get("x") is None
            or later.get("y") is None
        ):
            continue
        midpoint = (
            (float(previous["x"]) + float(later["x"])) / 2.0,
            (float(previous["y"]) + float(later["y"])) / 2.0,
        )
        residuals.append(math.dist((float(current["x"]), float(current["y"])), midpoint))
    return residuals


def _confidence_weight(points: dict) -> float:
    confidences = [
        float(point.get("confidence", 0.0))
        for point in points.values()
        if point.get("x") is not None and point.get("y") is not None
    ]
    return mean_or_zero(confidences) if confidences else 0.0


def _annotation_key(annotation: dict) -> str:
    return f"{annotation.get('clip_id', '')}:{annotation.get('source_view', '')}:{annotation.get('frame_index', 0)}"


def _prediction_key(prediction: dict) -> str:
    return f"{prediction.get('clip_id', '')}:{prediction.get('source_view', '')}:{prediction.get('frame_index', 0)}"
