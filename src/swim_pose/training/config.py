from __future__ import annotations

from pathlib import Path

from ..io import ensure_parent, load_toml


def load_config(path: str | Path) -> dict:
    config = load_toml(path)
    for section in ("experiment", "dataset", "model", "training"):
        if section not in config:
            raise ValueError(f"Missing required config section: {section}")
    return config


def experiment_output_dir(config: dict) -> Path:
    output_dir = config["experiment"]["output_dir"]
    return ensure_parent(Path(output_dir) / "placeholder.txt").parent

