from __future__ import annotations

from pathlib import Path

from ..io import ensure_parent, load_toml
from ..pathing import require_repo_root, resolve_repo_managed_path


CONFIG_PATH_FIELDS = {
    "experiment": ("output_dir",),
    "dataset": (
        "train_index",
        "val_index",
        "annotation_index",
        "labeled_index",
        "unlabeled_index",
        "image_root",
        "video_index",
    ),
    "model": ("pretrained_checkpoint",),
    "bridge": ("teacher_checkpoint", "context_index"),
}


def load_config(path: str | Path) -> dict:
    repo_root = require_repo_root()
    config = load_toml(resolve_repo_managed_path(path, repo_root))
    for section in ("experiment", "dataset", "model", "training"):
        if section not in config:
            raise ValueError(f"Missing required config section: {section}")
    for section, keys in CONFIG_PATH_FIELDS.items():
        values = config.get(section, {})
        for key in keys:
            value = values.get(key, "")
            if value:
                values[key] = str(resolve_repo_managed_path(value, repo_root))
    return config


def experiment_output_dir(config: dict) -> Path:
    output_dir = config["experiment"]["output_dir"]
    return ensure_parent(Path(output_dir) / "placeholder.txt").parent
