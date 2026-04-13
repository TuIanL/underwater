from __future__ import annotations

import random
from pathlib import Path

import numpy as np


def set_random_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
    except ImportError:
        return
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_device(preferred: str | None = None) -> str:
    try:
        import torch
    except ImportError as exc:
        raise RuntimeError("PyTorch is required for training commands.") from exc
    if preferred:
        normalized = preferred.lower()
        if normalized == "cpu":
            return "cpu"
        if normalized == "cuda":
            if torch.cuda.is_available():
                return "cuda"
            raise RuntimeError("Requested cuda device, but CUDA is not available.")
        if normalized == "mps":
            if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                return "mps"
            raise RuntimeError("Requested mps device, but MPS is not available.")
        raise RuntimeError(f"Unsupported device override: {preferred}")
    if torch.cuda.is_available():
        return "cuda"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def checkpoint_path(output_dir: str | Path, name: str = "best.pt") -> Path:
    path = Path(output_dir) / name
    path.parent.mkdir(parents=True, exist_ok=True)
    return path
