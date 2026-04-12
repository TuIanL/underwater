from __future__ import annotations

from pathlib import Path

from ..io import read_jsonl, write_jsonl


def generate_pseudolabel_file(predictions_path: str | Path, output_path: str | Path, threshold: float) -> Path:
    predictions = read_jsonl(predictions_path)
    pseudolabels: list[dict] = []
    for row in predictions:
        points = {}
        for name, point in row["points"].items():
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
                },
            }
        )
    write_jsonl(output_path, pseudolabels)
    return Path(output_path)

