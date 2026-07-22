from __future__ import annotations

from pathlib import Path
from typing import Any

import torch


def save_checkpoint(
    path: str | Path,
    model: Any,
    optimizer: Any = None,
    step: int = 0,
    tokens_seen: int = 0,
    metrics: dict[str, float] | None = None,
) -> None:
    """Save model checkpoint state to disk."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    state = {
        "model": model.state_dict() if hasattr(model, "state_dict") else model,
        "optimizer": optimizer.state_dict() if optimizer is not None and hasattr(optimizer, "state_dict") else None,
        "step": step,
        "tokens_seen": tokens_seen,
        "metrics": metrics or {},
    }
    torch.save(state, path)


def load_checkpoint(
    path: str | Path,
    model: Any,
    optimizer: Any = None,
    map_location: str | torch.device = "cpu",
) -> dict[str, Any]:
    """Load model checkpoint state from disk."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Checkpoint file not found: {path}")
    state = torch.load(path, map_location=map_location, weights_only=False)
    if hasattr(model, "load_state_dict"):
        model.load_state_dict(state["model"])
    if optimizer is not None and state.get("optimizer") is not None and hasattr(optimizer, "load_state_dict"):
        optimizer.load_state_dict(state["optimizer"])
    return state


class CheckpointManager:
    """Manage lifecycle of training checkpoints including best model tracking and retention limits."""

    def __init__(
        self,
        checkpoint_dir: str | Path = "checkpoints",
        max_to_keep: int = 5,
        metric_name: str = "val_loss",
        mode: str = "min",
    ) -> None:
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.max_to_keep = max_to_keep
        self.metric_name = metric_name
        self.mode = mode
        self.best_metric = float("inf") if mode == "min" else float("-inf")
        self.saved_checkpoints: list[Path] = []

    def save(
        self,
        step: int,
        model: Any,
        optimizer: Any = None,
        tokens_seen: int = 0,
        metrics: dict[str, float] | None = None,
    ) -> Path:
        """Save a step checkpoint and update latest/best tracking."""
        ckpt_path = self.checkpoint_dir / f"checkpoint_step_{step}.pt"
        save_checkpoint(ckpt_path, model, optimizer, step, tokens_seen, metrics)
        self.saved_checkpoints.append(ckpt_path)

        # Update latest pointer/symlink
        latest_path = self.checkpoint_dir / "latest.pt"
        save_checkpoint(latest_path, model, optimizer, step, tokens_seen, metrics)

        # Track best model if metrics provided
        metrics = metrics or {}
        if self.metric_name in metrics:
            val = metrics[self.metric_name]
            is_best = (val < self.best_metric) if self.mode == "min" else (val > self.best_metric)
            if is_best:
                self.best_metric = val
                best_path = self.checkpoint_dir / "best.pt"
                save_checkpoint(best_path, model, optimizer, step, tokens_seen, metrics)

        # Prune old checkpoints if over max_to_keep
        while len(self.saved_checkpoints) > self.max_to_keep:
            oldest = self.saved_checkpoints.pop(0)
            if oldest.exists() and oldest.name not in ("latest.pt", "best.pt"):
                oldest.unlink()

        return ckpt_path

    def load_latest(self, model: Any, optimizer: Any = None, map_location: str = "cpu") -> dict[str, Any]:
        """Load the latest checkpoint."""
        latest_path = self.checkpoint_dir / "latest.pt"
        return load_checkpoint(latest_path, model, optimizer, map_location=map_location)

    def load_best(self, model: Any, optimizer: Any = None, map_location: str = "cpu") -> dict[str, Any]:
        """Load the best checkpoint."""
        best_path = self.checkpoint_dir / "best.pt"
        return load_checkpoint(best_path, model, optimizer, map_location=map_location)
