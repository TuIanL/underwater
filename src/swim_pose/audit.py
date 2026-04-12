from __future__ import annotations

import math
from pathlib import Path

from .annotations import validate_annotation
from .io import read_json, write_json

PAIR_KEYS = [
    ("left_shoulder", "right_shoulder"),
    ("left_elbow", "right_elbow"),
    ("left_wrist", "right_wrist"),
    ("left_hip", "right_hip"),
    ("left_knee", "right_knee"),
    ("left_ankle", "right_ankle"),
    ("left_heel", "right_heel"),
    ("left_toe", "right_toe"),
]


def audit_annotations(annotation_root: str | Path, output_path: str | Path) -> Path:
    root = Path(annotation_root)
    summary = {
        "files": 0,
        "files_with_warnings": 0,
        "files_with_notes": 0,
        "warnings": [],
        "notes": [],
        "totals": {
            "all_zero_files": 0,
            "pending_files": 0,
            "no_swimmer_files": 0,
            "ankle_heel_overlap": 0,
            "ankle_toe_overlap": 0,
            "duplicate_left_right_points": 0,
            "validation_errors": 0,
        },
    }

    for annotation_path in sorted(root.rglob("*.json")):
        data = read_json(annotation_path)
        summary["files"] += 1
        issues = _audit_file(annotation_path, data)
        warning_issues = [issue for issue in issues if issue["severity"] == "warning"]
        note_issues = [issue for issue in issues if issue["severity"] == "note"]
        if warning_issues:
            summary["files_with_warnings"] += 1
            summary["warnings"].append(
                {
                    "file": str(annotation_path),
                    "issues": warning_issues,
                }
            )
        if note_issues:
            summary["files_with_notes"] += 1
            summary["notes"].append(
                {
                    "file": str(annotation_path),
                    "issues": note_issues,
                }
            )
        for issue in issues:
            kind = issue["kind"]
            if kind in summary["totals"]:
                summary["totals"][kind] += 1
    write_json(output_path, summary)
    return Path(output_path)


def _audit_file(annotation_path: Path, data: dict) -> list[dict]:
    warnings: list[dict] = []
    validation_errors = validate_annotation(data)
    if validation_errors:
        warnings.append({"kind": "validation_errors", "details": validation_errors, "severity": "warning"})
        return warnings

    frame_status = data.get("metadata", {}).get("frame_status", "pending")
    visible_count = 0
    for point in data["points"].values():
        if int(point.get("visibility", 0)) > 0:
            visible_count += 1
    if visible_count == 0:
        if frame_status == "no_swimmer":
            warnings.append(
                {
                    "kind": "no_swimmer_files",
                    "details": ["Frame explicitly marked as no swimmer"],
                    "severity": "note",
                }
            )
            return warnings
        if frame_status == "pending":
            warnings.append(
                {
                    "kind": "pending_files",
                    "details": ["Frame still pending labeling"],
                    "severity": "note",
                }
            )
            return warnings
        warnings.append({"kind": "all_zero_files", "details": ["No labeled keypoints in file"], "severity": "warning"})

    for side in ("left", "right"):
        ankle = data["points"][f"{side}_ankle"]
        heel = data["points"][f"{side}_heel"]
        toe = data["points"][f"{side}_toe"]
        if _distance_if_visible(ankle, heel) is not None and _distance_if_visible(ankle, heel) < 3.0:
            warnings.append(
                {
                    "kind": "ankle_heel_overlap",
                    "details": [f"{side} heel too close to ankle"],
                    "severity": "warning",
                }
            )
        if _distance_if_visible(ankle, toe) is not None and _distance_if_visible(ankle, toe) < 3.0:
            warnings.append(
                {
                    "kind": "ankle_toe_overlap",
                    "details": [f"{side} toe too close to ankle"],
                    "severity": "warning",
                }
            )

    for left_name, right_name in PAIR_KEYS:
        left_point = data["points"][left_name]
        right_point = data["points"][right_name]
        if _distance_if_visible(left_point, right_point) == 0.0:
            warnings.append(
                {
                    "kind": "duplicate_left_right_points",
                    "details": [f"{left_name} and {right_name} share identical coordinates"],
                    "severity": "warning",
                }
            )

    return warnings


def _distance_if_visible(first: dict, second: dict) -> float | None:
    if int(first.get("visibility", 0)) == 0 or int(second.get("visibility", 0)) == 0:
        return None
    if first.get("x") is None or first.get("y") is None or second.get("x") is None or second.get("y") is None:
        return None
    return math.dist((float(first["x"]), float(first["y"])), (float(second["x"]), float(second["y"])))
