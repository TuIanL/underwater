from __future__ import annotations

import json
import math
from collections import defaultdict
from pathlib import Path

from ..constants import KEYPOINT_NAMES
from ..io import read_csv_rows, read_json, read_jsonl, write_json
from ..pathing import find_repo_root, resolve_repo_managed_path


def evaluate_predictions_file(predictions_path: str | Path, annotations_path: str | Path, output_path: str | Path) -> Path:
    predictions = read_jsonl(predictions_path)
    annotation_lookup = load_annotation_lookup(annotations_path)

    aggregate_errors: list[float] = []
    visible_errors: list[float] = []
    occluded_errors: list[float] = []
    per_joint = defaultdict(list)
    pck_005 = []
    pck_010 = []
    temporal_groups = defaultdict(list)

    for prediction in predictions:
        annotation = resolve_annotation(annotation_lookup, prediction)
        if annotation is None:
            continue
        normalizer = annotation_scale(annotation)
        for keypoint in KEYPOINT_NAMES:
            pred_point = prediction["points"][keypoint]
            gt_point = annotation["points"][keypoint]
            if gt_point["visibility"] == 0:
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
        temporal_groups[(prediction["clip_id"], prediction["source_view"])].append(prediction)

    report = {
        "overall": {
            "mean_normalized_error": mean_or_zero(aggregate_errors),
            "pck@0.05": mean_or_zero(pck_005),
            "pck@0.10": mean_or_zero(pck_010),
            "visible_mean_error": mean_or_zero(visible_errors),
            "occluded_mean_error": mean_or_zero(occluded_errors),
            "temporal_jitter": mean_or_zero(collect_temporal_jitter(temporal_groups)),
        },
        "per_joint": {
            keypoint: {
                "mean_normalized_error": mean_or_zero(errors),
                "count": len(errors),
            }
            for keypoint, errors in per_joint.items()
        },
    }
    write_json(output_path, report)
    return Path(output_path)


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


def collect_temporal_jitter(groups: dict[tuple[str, str], list[dict]]) -> list[float]:
    jitters: list[float] = []
    for group_predictions in groups.values():
        ordered = sorted(group_predictions, key=lambda row: int(row["frame_index"]))
        for earlier, later in zip(ordered, ordered[1:]):
            for keypoint in KEYPOINT_NAMES:
                first = earlier["points"][keypoint]
                second = later["points"][keypoint]
                jitters.append(math.dist((first["x"], first["y"]), (second["x"], second["y"])))
    return jitters


def mean_or_zero(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def _annotation_key(annotation: dict) -> str:
    return f"{annotation.get('clip_id', '')}:{annotation.get('source_view', '')}:{annotation.get('frame_index', 0)}"


def _prediction_key(prediction: dict) -> str:
    return f"{prediction.get('clip_id', '')}:{prediction.get('source_view', '')}:{prediction.get('frame_index', 0)}"
