from __future__ import annotations

import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path

from .io import read_csv_rows, write_csv_rows
from .pathing import require_repo_root, resolve_persisted_source_path, serialize_workspace_path

VIDEO_EXTENSIONS = {".mp4", ".mov", ".avi", ".mkv"}
MANIFEST_FIELDS = [
    "clip_id",
    "athlete_id",
    "session_id",
    "raw_above_path",
    "raw_under_path",
    "stitched_path",
    "primary_view",
    "sync_status",
    "sync_offset_ms",
    "fps_above",
    "fps_under",
    "frame_count_above",
    "frame_count_under",
    "duration_above_s",
    "duration_under_s",
    "notes",
]

_VIEW_MARKERS = {
    "above": ("_above", "-above", "_top", "-top", "_surface", "-surface"),
    "under": ("_under", "-under", "_underwater", "-underwater"),
    "stitched": ("_stitched", "-stitched", "_dual", "-dual", "_preview", "-preview"),
}


@dataclass
class ClipManifestEntry:
    clip_id: str
    athlete_id: str
    session_id: str
    raw_above_path: str = ""
    raw_under_path: str = ""
    stitched_path: str = ""
    primary_view: str = ""
    sync_status: str = "pending_audit"
    sync_offset_ms: str = ""
    fps_above: str = ""
    fps_under: str = ""
    frame_count_above: str = ""
    frame_count_under: str = ""
    duration_above_s: str = ""
    duration_under_s: str = ""
    notes: str = ""

    def to_row(self) -> dict[str, str]:
        return {field: str(getattr(self, field, "")) for field in MANIFEST_FIELDS}


def discover_manifest(video_root: str | Path) -> list[ClipManifestEntry]:
    root = Path(video_root)
    repo_root = require_repo_root()
    grouped: dict[str, ClipManifestEntry] = {}
    for path in sorted(root.rglob("*")):
        if not path.is_file() or path.suffix.lower() not in VIDEO_EXTENSIONS:
            continue
        view = classify_view(path.stem)
        athlete_id, session_id = infer_ids(root, path, infer_clip_id(path.stem))
        clip_id = f"{athlete_id}_{session_id}_{infer_clip_id(path.stem)}"
        entry = grouped.setdefault(
            clip_id,
            ClipManifestEntry(
                clip_id=clip_id,
                athlete_id=athlete_id,
                session_id=session_id,
            ),
        )
        resolved = serialize_workspace_path(path, repo_root)
        if view == "above":
            entry.raw_above_path = resolved
        elif view == "under":
            entry.raw_under_path = resolved
        else:
            entry.stitched_path = resolved
        entry.primary_view = determine_primary_view(entry)
    return list(grouped.values())


def write_manifest(path: str | Path, entries: list[ClipManifestEntry]) -> Path:
    write_csv_rows(path, MANIFEST_FIELDS, (entry.to_row() for entry in entries))
    return Path(path)


def read_manifest(path: str | Path) -> list[dict[str, str]]:
    return read_csv_rows(path)


def audit_manifest(path: str | Path) -> list[dict[str, str]]:
    rows = read_manifest(path)
    audited: list[dict[str, str]] = []
    for row in rows:
        above = row.get("raw_above_path", "")
        under = row.get("raw_under_path", "")
        above_meta = probe_video(above) if above else {}
        under_meta = probe_video(under) if under else {}

        row["fps_above"] = stringify(above_meta.get("fps", row.get("fps_above", "")))
        row["fps_under"] = stringify(under_meta.get("fps", row.get("fps_under", "")))
        row["frame_count_above"] = stringify(above_meta.get("frames", row.get("frame_count_above", "")))
        row["frame_count_under"] = stringify(under_meta.get("frames", row.get("frame_count_under", "")))
        row["duration_above_s"] = stringify(above_meta.get("duration_s", row.get("duration_above_s", "")))
        row["duration_under_s"] = stringify(under_meta.get("duration_s", row.get("duration_under_s", "")))

        row["primary_view"] = determine_primary_view(
            ClipManifestEntry(
                clip_id=row.get("clip_id", ""),
                athlete_id=row.get("athlete_id", ""),
                session_id=row.get("session_id", ""),
                raw_above_path=above,
                raw_under_path=under,
                stitched_path=row.get("stitched_path", ""),
            )
        )
        row["sync_status"] = classify_sync_status(row, above_meta, under_meta)
        audited.append(row)
    return audited


def migrate_manifest_paths(
    path: str | Path,
    output_path: str | Path | None = None,
    legacy_base: str | Path | None = None,
) -> tuple[Path, dict[str, int]]:
    repo_root = require_repo_root()
    rows = read_manifest(path)
    fieldnames = list(rows[0].keys()) if rows else MANIFEST_FIELDS
    legacy_root = Path(legacy_base).expanduser().resolve() if legacy_base else None
    migrated_rows: list[dict[str, str]] = []
    updated_fields = 0

    for row in rows:
        migrated_row = dict(row)
        for field in ("raw_above_path", "raw_under_path", "stitched_path"):
            original = row.get(field, "").strip()
            if not original:
                continue
            migrated = _migrate_manifest_field(original, repo_root=repo_root, legacy_root=legacy_root)
            if migrated != original:
                updated_fields += 1
            migrated_row[field] = migrated
        migrated_rows.append(migrated_row)

    destination = Path(output_path) if output_path is not None else Path(path)
    write_csv_rows(destination, fieldnames, migrated_rows)
    return Path(destination), {"rows": len(rows), "updated_fields": updated_fields}


def classify_view(stem: str) -> str:
    normalized = stem.lower()
    for view, markers in _VIEW_MARKERS.items():
        if normalized.endswith(markers):
            return view
    return "stitched"


def infer_clip_id(stem: str) -> str:
    normalized = stem
    lowered = stem.lower()
    for markers in _VIEW_MARKERS.values():
        for marker in markers:
            if lowered.endswith(marker):
                return normalized[: -len(marker)]
    return normalized


def infer_ids(video_root: Path, path: Path, fallback_clip_id: str) -> tuple[str, str]:
    relative_parts = path.relative_to(video_root).parts
    athlete_id = relative_parts[0] if len(relative_parts) >= 2 else fallback_clip_id.split("_")[0]
    session_id = relative_parts[1] if len(relative_parts) >= 3 else "session_unknown"
    return athlete_id, session_id


def determine_primary_view(entry: ClipManifestEntry) -> str:
    if entry.stitched_path:
        return "stitched"
    if entry.raw_above_path and entry.raw_under_path:
        return "dual_raw_only"
    if entry.raw_under_path:
        return "under_only"
    if entry.raw_above_path:
        return "above_only"
    return "missing"


def probe_video(path: str) -> dict[str, float | int]:
    if not path:
        return {}
    resolved = resolve_persisted_source_path(path)
    if not resolved.exists():
        return {}

    ffprobe = shutil.which("ffprobe")
    if not ffprobe:
        return _probe_video_with_opencv(resolved)

    command = [
        ffprobe,
        "-v",
        "error",
        "-count_frames",
        "-select_streams",
        "v:0",
        "-show_entries",
        "stream=nb_read_frames,avg_frame_rate",
        "-show_entries",
        "format=duration",
        "-of",
        "default=noprint_wrappers=1:nokey=0",
        str(resolved),
    ]
    try:
        result = subprocess.run(command, check=True, capture_output=True, text=True)
    except subprocess.CalledProcessError:
        return _probe_video_with_opencv(resolved)

    metadata: dict[str, float | int] = {}
    for line in result.stdout.splitlines():
        if not line.strip() or "=" not in line:
            continue
        key, value = line.split("=", 1)
        if key == "duration":
            metadata["duration_s"] = round(float(value), 3)
        if key == "nb_read_frames" and value.isdigit():
            metadata["frames"] = int(value)
        if key == "avg_frame_rate" and "/" in value:
            numerator, denominator = value.split("/", 1)
            if denominator != "0":
                metadata["fps"] = round(float(numerator) / float(denominator), 3)
    if not metadata:
        return _probe_video_with_opencv(resolved)
    return metadata


def classify_sync_status(
    row: dict[str, str],
    above_meta: dict[str, float | int],
    under_meta: dict[str, float | int],
) -> str:
    if not row.get("stitched_path"):
        return "missing_stitched"
    if not row.get("raw_above_path") and not row.get("raw_under_path"):
        return "stitched_only"
    if not row.get("raw_above_path") or not row.get("raw_under_path"):
        return "partial_raw_provenance"
    if not above_meta or not under_meta:
        return "manual_review_required"
    duration_delta = abs(float(above_meta.get("duration_s", 0.0)) - float(under_meta.get("duration_s", 0.0)))
    frame_delta = abs(int(above_meta.get("frames", 0)) - int(under_meta.get("frames", 0)))
    if duration_delta <= 0.1 and frame_delta <= 3:
        return "aligned_optional_raw"
    return "duration_mismatch_optional_raw"


def stringify(value: object) -> str:
    if value is None:
        return ""
    return str(value)


def _probe_video_with_opencv(path: Path) -> dict[str, float | int]:
    try:
        import cv2
    except ImportError:
        return {}

    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        return {}
    frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    fps = float(capture.get(cv2.CAP_PROP_FPS) or 0.0)
    duration = round(frame_count / fps, 3) if fps > 0 else 0.0
    capture.release()
    metadata: dict[str, float | int] = {}
    if frame_count > 0:
        metadata["frames"] = frame_count
    if fps > 0:
        metadata["fps"] = round(fps, 3)
    if duration > 0:
        metadata["duration_s"] = duration
    return metadata


def _migrate_manifest_field(value: str, repo_root: Path, legacy_root: Path | None) -> str:
    original = Path(value).expanduser()
    if original.is_absolute():
        return serialize_workspace_path(original, repo_root)

    repo_candidate = (repo_root / original).resolve()
    if repo_candidate.exists():
        return serialize_workspace_path(repo_candidate, repo_root)

    if legacy_root is not None:
        legacy_candidate = (legacy_root / original).resolve()
        if legacy_candidate.exists():
            return serialize_workspace_path(legacy_candidate, repo_root)

    raise FileNotFoundError(
        f"Could not migrate relative manifest path '{value}'. "
        "It does not resolve under the repository root"
        + ("" if legacy_root is None else f" or legacy base '{legacy_root}'")
        + "."
    )
