from __future__ import annotations

import random
from pathlib import Path

from .io import read_csv_rows, write_csv_rows
from .manifest import probe_video

SEED_FIELDS = [
    "clip_id",
    "athlete_id",
    "session_id",
    "source_view",
    "frame_index",
    "selection_reason",
]


def select_seed_frames(
    manifest_path: str | Path,
    output_path: str | Path,
    frames_per_clip: int = 12,
    source_view: str = "stitched",
    seed: int = 7,
) -> Path:
    rows = read_csv_rows(manifest_path)
    rng = random.Random(seed)
    selected: list[dict[str, object]] = []
    phase_positions = [0.08, 0.2, 0.32, 0.44, 0.56, 0.68, 0.8, 0.92]
    for row in rows:
        frame_count = _resolve_frame_count(row, source_view)
        if frame_count <= 0:
            continue
        note_text = row.get("notes", "").lower()
        hard_tags = [tag for tag in ("waterline", "splash", "overlap", "reflection") if tag in note_text]
        positions = phase_positions[: min(frames_per_clip, len(phase_positions))]
        while len(positions) < frames_per_clip:
            positions.append(rng.random())
        for index, position in enumerate(sorted(positions[:frames_per_clip])):
            selected.append(
                {
                    "clip_id": row.get("clip_id", ""),
                    "athlete_id": row.get("athlete_id", ""),
                    "session_id": row.get("session_id", ""),
                    "source_view": source_view,
                    "frame_index": min(frame_count - 1, round(position * (frame_count - 1))),
                    "selection_reason": _selection_reason(index, hard_tags),
                }
            )
    write_csv_rows(output_path, SEED_FIELDS, selected)
    return Path(output_path)


def create_group_splits(
    index_path: str | Path,
    output_dir: str | Path,
    group_by: tuple[str, ...] = ("athlete_id", "session_id"),
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    seed: int = 7,
) -> list[Path]:
    rows = read_csv_rows(index_path)
    grouped: dict[tuple[str, ...], list[dict[str, str]]] = {}
    for row in rows:
        key = tuple(row.get(field, "") for field in group_by)
        grouped.setdefault(key, []).append(row)

    groups = list(grouped.items())
    rng = random.Random(seed)
    rng.shuffle(groups)

    total = len(groups)
    test_cutoff = round(total * test_ratio)
    val_cutoff = round(total * val_ratio)

    partitions = {
        "test": groups[:test_cutoff],
        "val": groups[test_cutoff:test_cutoff + val_cutoff],
        "train": groups[test_cutoff + val_cutoff:],
    }

    output_paths: list[Path] = []
    output_root = Path(output_dir)
    output_root.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys()) if rows else []
    for split_name, split_groups in partitions.items():
        split_rows = [row for _, grouped_rows in split_groups for row in grouped_rows]
        destination = output_root / f"{split_name}.csv"
        write_csv_rows(destination, fieldnames, split_rows)
        output_paths.append(destination)
    return output_paths


def _resolve_frame_count(row: dict[str, str], source_view: str) -> int:
    if source_view == "above":
        value = row.get("frame_count_above", "")
        if value.isdigit():
            return int(value)
        return int(probe_video(row.get("raw_above_path", "")).get("frames", 0))

    if source_view == "under":
        value = row.get("frame_count_under", "") or row.get("frame_count_above", "")
        if value.isdigit():
            return int(value)
        fallback = row.get("raw_under_path", "") or row.get("raw_above_path", "")
        return int(probe_video(fallback).get("frames", 0))

    stitched_frames = probe_video(row.get("stitched_path", "")).get("frames", 0)
    return int(stitched_frames)


def _selection_reason(index: int, hard_tags: list[str]) -> str:
    anchor = f"cycle_anchor_{index + 1}"
    if hard_tags:
        return f"{anchor}|targeted:{'|'.join(hard_tags)}"
    return anchor
