from __future__ import annotations

from pathlib import Path

from ..io import read_jsonl, write_jsonl


def generate_pseudolabel_file(
    predictions_path: str | Path,
    output_path: str | Path,
    threshold: float,
    use_filtered: bool = False,
) -> Path:
    predictions = read_jsonl(predictions_path)
    pseudolabels: list[dict] = []
    for row in predictions:
        points_source = row.get("filtered_points") if use_filtered and isinstance(row.get("filtered_points"), dict) else row["points"]
        points = {}
        for name, point in points_source.items():
            accepted = float(point.get("confidence", 0.0)) >= threshold and int(point.get("visibility", 0)) > 0
            points[name] = {
                "x": point["x"] if accepted else None,
                "y": point["y"] if accepted else None,
                "visibility": point["visibility"] if accepted else 0,
                "confidence": point.get("confidence", 0.0),
            }
        pseudolabels.append(
            {
                "clip_id": row.get("clip_id", ""),
                "frame_index": row.get("frame_index", 0),
                "source_view": row.get("source_view", ""),
                "image_path": row.get("image_path", ""),
                "athlete_id": row.get("athlete_id", ""),
                "session_id": row.get("session_id", ""),
                "points": points,
                "metadata": {
                    "generated_from": str(predictions_path),
                    "threshold": threshold,
                    "use_filtered": use_filtered,
                },
            }
        )
    write_jsonl(output_path, pseudolabels)
    return Path(output_path)
