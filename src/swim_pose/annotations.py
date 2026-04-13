from __future__ import annotations

from pathlib import Path

from .constants import FRAME_STATUSES, KEYPOINT_NAMES, SOURCE_VIEWS, VISIBILITY_STATES
from .io import ensure_parent, read_csv_rows, read_json, write_csv_rows, write_json
from .pathing import find_repo_root, serialize_workspace_path


ANNOTATION_INDEX_FIELDS = [
    "annotation_path",
    "image_path",
    "clip_id",
    "athlete_id",
    "session_id",
    "frame_index",
    "source_view",
    "frame_status",
    "difficulties",
]


def build_template() -> dict:
    points = {
        keypoint: {"x": None, "y": None, "visibility": 0}
        for keypoint in KEYPOINT_NAMES
    }
    return {
        "clip_id": "",
        "frame_index": 0,
        "source_view": "under",
        "image_path": "",
        "athlete_id": "",
        "session_id": "",
        "points": points,
        "metadata": {
            "frame_status": "pending",
            "stroke_phase": "",
            "difficulties": [],
        },
    }


def validate_annotation(data: dict) -> list[str]:
    errors: list[str] = []
    if data.get("source_view") not in SOURCE_VIEWS:
        errors.append("source_view must be one of: above, under, stitched")
    metadata = data.get("metadata", {})
    if not isinstance(metadata, dict):
        errors.append("metadata must be an object")
        metadata = {}
    frame_status = metadata.get("frame_status", "pending")
    if frame_status not in FRAME_STATUSES:
        errors.append(f"metadata.frame_status must be one of: {', '.join(FRAME_STATUSES)}")

    points = data.get("points")
    if not isinstance(points, dict):
        errors.append("points must be an object keyed by keypoint name")
        return errors

    missing = [name for name in KEYPOINT_NAMES if name not in points]
    unexpected = [name for name in points if name not in KEYPOINT_NAMES]
    if missing:
        errors.append(f"missing keypoints: {', '.join(missing)}")
    if unexpected:
        errors.append(f"unexpected keypoints: {', '.join(unexpected)}")

    for name in KEYPOINT_NAMES:
        point = points.get(name)
        if not isinstance(point, dict):
            errors.append(f"{name} must be an object")
            continue
        visibility = point.get("visibility")
        if visibility not in VISIBILITY_STATES:
            errors.append(f"{name}.visibility must be one of 0, 1, 2")
            continue
        x = point.get("x")
        y = point.get("y")
        if visibility in {1, 2} and not _is_number(x, y):
            errors.append(f"{name} requires numeric x/y when visibility is {visibility}")
        if visibility == 0 and (x is not None or y is not None) and not _is_number(x, y):
            errors.append(f"{name} uses invalid coordinates for visibility 0")

    if frame_status == "no_swimmer":
        invalid_points = [
            name
            for name in KEYPOINT_NAMES
            if _point_is_marked(points.get(name, {}))
        ]
        if invalid_points:
            errors.append(
                "metadata.frame_status=no_swimmer requires every keypoint to stay empty; "
                f"found labeled content for: {', '.join(invalid_points)}"
            )
    return errors


def write_template(path: str | Path) -> Path:
    destination = ensure_parent(path)
    write_json(destination, build_template())
    return destination


def validate_file(path: str | Path) -> list[str]:
    return validate_annotation(read_json(path))


def build_annotation_index(annotation_root: str | Path, output_path: str | Path) -> Path:
    root = Path(annotation_root)
    repo_root = find_repo_root(root)
    rows: list[dict[str, object]] = []
    for annotation_path in sorted(root.rglob("*.json")):
        data = read_json(annotation_path)
        errors = validate_annotation(data)
        if errors:
            raise ValueError(f"{annotation_path} failed validation: {'; '.join(errors)}")
        metadata = data.get("metadata", {})
        frame_status = metadata.get("frame_status", "pending")
        if frame_status != "labeled":
            continue
        rows.append(
            {
                "annotation_path": serialize_workspace_path(annotation_path, repo_root),
                "image_path": data.get("image_path", ""),
                "clip_id": data.get("clip_id", ""),
                "athlete_id": data.get("athlete_id", ""),
                "session_id": data.get("session_id", ""),
                "frame_index": data.get("frame_index", 0),
                "source_view": data.get("source_view", ""),
                "frame_status": frame_status,
                "difficulties": "|".join(metadata.get("difficulties", [])),
            }
        )
    write_csv_rows(output_path, ANNOTATION_INDEX_FIELDS, rows)
    return Path(output_path)


def scaffold_annotations(seed_csv: str | Path, frame_root: str | Path, output_root: str | Path) -> list[Path]:
    seed_rows = read_csv_rows(seed_csv)
    frame_root_path = Path(frame_root)
    output_root_path = Path(output_root)
    created: list[Path] = []
    for row in seed_rows:
        annotation = build_template()
        clip_id = row.get("clip_id", "")
        athlete_id = row.get("athlete_id", "")
        session_id = row.get("session_id", "")
        source_view = row.get("source_view", "under")
        frame_index = _safe_int(row.get("frame_index", 0))
        image_path = (
            Path(athlete_id)
            / session_id
            / clip_id
            / source_view
            / f"{clip_id}_{source_view}_{frame_index:06d}.jpg"
        )
        annotation["clip_id"] = clip_id
        annotation["frame_index"] = frame_index
        annotation["source_view"] = source_view
        annotation["image_path"] = str(image_path if (frame_root_path / image_path).exists() else image_path)
        annotation["athlete_id"] = athlete_id
        annotation["session_id"] = session_id
        annotation["metadata"]["frame_status"] = "pending"
        annotation["metadata"]["selection_reason"] = row.get("selection_reason", "")
        destination = output_root_path / athlete_id / session_id / clip_id / source_view / f"{clip_id}_{source_view}_{frame_index:06d}.json"
        write_json(destination, annotation)
        created.append(destination)
    return created


def _is_number(*values: object) -> bool:
    return all(isinstance(value, (int, float)) for value in values)


def _safe_int(value: object) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return 0


def _point_is_marked(point: dict) -> bool:
    visibility = point.get("visibility", 0)
    if visibility != 0:
        return True
    return point.get("x") is not None or point.get("y") is not None
