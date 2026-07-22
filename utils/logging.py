from __future__ import annotations

from datetime import datetime
import logging
from pathlib import Path
import sys
from typing import Any

from .environment import get_git_hash
from .file_utils import ensure_dir, write_json


def setup_logger(
    name: str = "vajra_lm",
    log_file: str | Path | None = None,
    level: int = logging.INFO,
) -> logging.Logger:
    """Setup structured logging to console and optional log file."""
    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.handlers.clear()

    formatter = logging.Formatter(
        fmt="[%(asctime)s] [%(levelname)s] [%(name)s]: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # Console handler
    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)

    # File handler
    if log_file:
        log_path = Path(log_file)
        ensure_dir(log_path.parent)
        file_handler = logging.FileHandler(log_path, encoding="utf-8")
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger


def create_experiment_dir(
    base_dir: str | Path = "runs",
    name: str = "exp",
    config: dict[str, Any] | None = None,
) -> Path:
    """Create a unique timestamped experiment directory and record metadata."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    exp_dir = ensure_dir(Path(base_dir) / f"{name}_{timestamp}")

    metadata = {
        "experiment_name": name,
        "timestamp": timestamp,
        "git_hash": get_git_hash(),
        "python_version": sys.version,
    }

    if config:
        write_json(config, exp_dir / "config.json")
    write_json(metadata, exp_dir / "metadata.json")

    return exp_dir
