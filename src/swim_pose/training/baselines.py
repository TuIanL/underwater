from __future__ import annotations


LEGACY_HEATMAP_BASELINE = "legacy_heatmap"
YOLO_POSE_BASELINE = "yolo_pose"


def model_family(config: dict) -> str:
    model_config = config.get("model", {})
    family = str(model_config.get("family", LEGACY_HEATMAP_BASELINE)).strip().lower()
    return family or LEGACY_HEATMAP_BASELINE


def is_yolo_pose_config(config: dict) -> bool:
    return model_family(config) == YOLO_POSE_BASELINE
