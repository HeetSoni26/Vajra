from __future__ import annotations

import os
import random
import subprocess
from typing import Any

import numpy as np
import torch


def set_seed(seed: int = 42, deterministic: bool = False) -> None:
    """Set random seed across random, numpy, and torch for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)

    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def get_device() -> torch.device:
    """Automatically detect and return the optimal available computing device."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def get_memory_info() -> dict[str, Any]:
    """Retrieve current GPU and system memory statistics."""
    info: dict[str, Any] = {}
    if torch.cuda.is_available():
        info["cuda_device_name"] = torch.cuda.get_device_name(0)
        info["cuda_allocated_mb"] = round(torch.cuda.memory_allocated(0) / (1024**2), 2)
        info["cuda_reserved_mb"] = round(torch.cuda.memory_reserved(0) / (1024**2), 2)
        info["cuda_max_allocated_mb"] = round(torch.cuda.max_memory_allocated(0) / (1024**2), 2)
    else:
        info["device"] = "cpu"
    return info


def get_git_hash() -> str | None:
    """Get current git commit hash if running in a git repository."""
    try:
        cmd = ["git", "rev-parse", "--short", "HEAD"]
        output = subprocess.check_output(cmd, stderr=subprocess.DEVNULL)
        return output.decode("utf-8").strip()
    except Exception:
        return None
