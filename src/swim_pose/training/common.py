from __future__ import annotations

from contextlib import contextmanager
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


@contextmanager
def singleton_batchnorm_eval(module):
    try:
        import torch
    except ImportError as exc:
        raise RuntimeError("PyTorch is required for training commands.") from exc

    batchnorm_states: list[tuple[torch.nn.modules.batchnorm._BatchNorm, bool]] = []
    for child in module.modules():
        if isinstance(child, torch.nn.modules.batchnorm._BatchNorm):
            batchnorm_states.append((child, child.training))
            if child.training:
                child.eval()
    try:
        yield
    finally:
        for child, was_training in batchnorm_states:
            child.train(was_training)


def forward_with_singleton_batch_support(module, inputs):
    if getattr(inputs, "shape", (0,))[0] != 1:
        return module(inputs)
    # BatchNorm cannot estimate statistics from a singleton batch once spatial
    # dimensions collapse to 1x1, so reuse running stats for that forward pass.
    with singleton_batchnorm_eval(module):
        return module(inputs)
