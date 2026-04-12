from __future__ import annotations

from pathlib import Path

from .io import read_csv_rows, write_csv_rows

UNLABELED_INDEX_FIELDS = [
    "image_path",
    "clip_id",
    "athlete_id",
    "session_id",
    "frame_index",
    "source_view",
]


def extract_frames_from_manifest(
    manifest_path: str | Path,
    output_root: str | Path,
    index_output: str | Path,
    views: tuple[str, ...] = ("stitched",),
    every_nth: int = 1,
) -> Path:
    try:
        import cv2
    except ImportError as exc:
        raise RuntimeError("opencv-python is required for frame extraction. Install dependencies first.") from exc

    rows = read_csv_rows(manifest_path)
    output_root_path = Path(output_root)
    index_rows: list[dict[str, object]] = []

    for row in rows:
        for view in views:
            video_path = _video_path_for_view(row, view)
            if not video_path:
                continue
            capture = cv2.VideoCapture(video_path)
            if not capture.isOpened():
                continue
            clip_id = row.get("clip_id", "")
            athlete_id = row.get("athlete_id", "")
            session_id = row.get("session_id", "")
            frame_index = 0
            while True:
                success, frame = capture.read()
                if not success:
                    break
                if frame_index % max(every_nth, 1) != 0:
                    frame_index += 1
                    continue
                relative_path = (
                    Path(athlete_id)
                    / session_id
                    / clip_id
                    / view
                    / f"{clip_id}_{view}_{frame_index:06d}.jpg"
                )
                destination = output_root_path / relative_path
                destination.parent.mkdir(parents=True, exist_ok=True)
                cv2.imwrite(str(destination), frame)
                index_rows.append(
                    {
                        "image_path": str(relative_path),
                        "clip_id": clip_id,
                        "athlete_id": athlete_id,
                        "session_id": session_id,
                        "frame_index": frame_index,
                        "source_view": view,
                    }
                )
                frame_index += 1
            capture.release()

    write_csv_rows(index_output, UNLABELED_INDEX_FIELDS, index_rows)
    return Path(index_output)


def _video_path_for_view(row: dict[str, str], view: str) -> str:
    if view == "above":
        return row.get("raw_above_path", "")
    if view == "under":
        return row.get("raw_under_path", "")
    if view == "stitched":
        return row.get("stitched_path", "")
    return ""
