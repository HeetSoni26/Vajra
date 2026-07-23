"""
Enhanced checkpoint management for Vajra production training.
Adds: JSON metadata sidecar, resume verification, disk-space guard,
      corrupted-checkpoint detection, and checkpoint registry.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import torch

from utils.logging import setup_logger

logger = setup_logger("checkpoint")

# ──────────────────────────────────────────────────────────────
# Low-level save / load helpers
# ──────────────────────────────────────────────────────────────

def save_checkpoint(
    path: str | Path,
    model: Any,
    optimizer: Any = None,
    step: int = 0,
    tokens_seen: int = 0,
    metrics: dict[str, float] | None = None,
    scheduler_state: dict | None = None,
    rng_state: dict | None = None,
) -> None:
    """Save a full training checkpoint (.pt) plus a JSON metadata sidecar."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    # Capture Python / CUDA / NumPy RNG states for full reproducibility
    if rng_state is None:
        rng_state = _capture_rng_state()

    state: dict[str, Any] = {
        "model": model.state_dict() if hasattr(model, "state_dict") else model,
        "optimizer": optimizer.state_dict() if optimizer is not None and hasattr(optimizer, "state_dict") else None,
        "scheduler": scheduler_state,
        "rng_state": rng_state,
        "step": step,
        "tokens_seen": tokens_seen,
        "metrics": metrics or {},
        "saved_at": time.strftime("%Y-%m-%dT%H:%M:%SZ"),
    }
    torch.save(state, path)

    # Write JSON metadata sidecar for quick inspection without loading the tensor file
    meta_path = path.with_suffix(".meta.json")
    meta: dict[str, Any] = {
        "step": step,
        "tokens_seen": tokens_seen,
        "metrics": metrics or {},
        "saved_at": state["saved_at"],
        "has_optimizer": optimizer is not None,
        "has_scheduler": scheduler_state is not None,
        "checkpoint_file": path.name,
        "size_bytes": path.stat().st_size,
    }
    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")


def load_checkpoint(
    path: str | Path,
    model: Any,
    optimizer: Any = None,
    map_location: str | torch.device = "cpu",
    strict: bool = True,
) -> dict[str, Any]:
    """Load a training checkpoint with corrupted-file detection."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {path}")

    try:
        state = torch.load(path, map_location=map_location, weights_only=False)
    except Exception as exc:
        raise RuntimeError(f"Checkpoint appears corrupted (failed to load): {path}\n{exc}") from exc

    if "model" not in state:
        raise RuntimeError(f"Checkpoint missing 'model' key — likely corrupted: {path}")

    if hasattr(model, "load_state_dict"):
        incompatible = model.load_state_dict(state["model"], strict=strict)
        if strict and (incompatible.missing_keys or incompatible.unexpected_keys):
            raise RuntimeError(
                f"State dict mismatch: missing={incompatible.missing_keys}, "
                f"unexpected={incompatible.unexpected_keys}"
            )

    if optimizer is not None and state.get("optimizer") is not None and hasattr(optimizer, "load_state_dict"):
        optimizer.load_state_dict(state["optimizer"])

    return state


# ──────────────────────────────────────────────────────────────
# RNG helpers
# ──────────────────────────────────────────────────────────────

def _capture_rng_state() -> dict:
    import random
    import numpy as np
    state: dict[str, Any] = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch_cpu": torch.get_rng_state(),
    }
    if torch.cuda.is_available():
        state["torch_cuda"] = torch.cuda.get_rng_state_all()
    return state


def _restore_rng_state(rng_state: dict) -> None:
    import random
    import numpy as np
    if "python" in rng_state:
        random.setstate(rng_state["python"])
    if "numpy" in rng_state:
        np.random.set_state(rng_state["numpy"])
    if "torch_cpu" in rng_state:
        torch.set_rng_state(rng_state["torch_cpu"])
    if "torch_cuda" in rng_state and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(rng_state["torch_cuda"])


# ──────────────────────────────────────────────────────────────
# CheckpointManager
# ──────────────────────────────────────────────────────────────

class CheckpointManager:
    """
    Lifecycle manager for Vajra training checkpoints.

    Features:
    - Per-step checkpoint files + JSON metadata sidecars
    - latest.pt / best.pt pointers
    - Automatic rotation (keeps max_to_keep most recent)
    - Checkpoint registry (checkpoints.json) for auditing
    - Corrupted-checkpoint detection on load
    - Disk-space guard before saving
    - Resume verification (validates step continuity)
    """

    def __init__(
        self,
        checkpoint_dir: str | Path = "checkpoints",
        max_to_keep: int = 5,
        metric_name: str = "val_loss",
        mode: str = "min",
        min_free_gb: float = 2.0,
    ) -> None:
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.max_to_keep = max_to_keep
        self.metric_name = metric_name
        self.mode = mode
        self.min_free_gb = min_free_gb
        self.best_metric = float("inf") if mode == "min" else float("-inf")
        self.saved_checkpoints: list[Path] = []

        # Load existing registry if resuming
        self._registry_path = self.checkpoint_dir / "checkpoints.json"
        self._registry: list[dict] = self._load_registry()

    # ── Registry ──────────────────────────────────────────────

    def _load_registry(self) -> list[dict]:
        if self._registry_path.exists():
            try:
                return json.loads(self._registry_path.read_text(encoding="utf-8"))
            except Exception:
                return []
        return []

    def _save_registry(self) -> None:
        self._registry_path.write_text(
            json.dumps(self._registry, indent=2), encoding="utf-8"
        )

    # ── Disk-space guard ──────────────────────────────────────

    def _check_disk_space(self) -> None:
        import shutil
        total, used, free = shutil.disk_usage(str(self.checkpoint_dir))
        free_gb = free / (1024 ** 3)
        if free_gb < self.min_free_gb:
            raise OSError(
                f"Insufficient disk space: {free_gb:.2f} GB free "
                f"(minimum required: {self.min_free_gb} GB). "
                f"Free space or reduce max_to_keep."
            )

    # ── Save ──────────────────────────────────────────────────

    def save(
        self,
        step: int,
        model: Any,
        optimizer: Any = None,
        tokens_seen: int = 0,
        metrics: dict[str, float] | None = None,
        scheduler_state: dict | None = None,
    ) -> Path:
        """Save a step checkpoint; update latest, best, and registry."""
        self._check_disk_space()

        ckpt_path = self.checkpoint_dir / f"checkpoint_step_{step}.pt"
        save_checkpoint(
            ckpt_path, model, optimizer, step, tokens_seen, metrics, scheduler_state
        )
        self.saved_checkpoints.append(ckpt_path)

        # Always update latest pointer
        latest_path = self.checkpoint_dir / "latest.pt"
        save_checkpoint(
            latest_path, model, optimizer, step, tokens_seen, metrics, scheduler_state
        )

        # Update best pointer when metric improves
        metrics = metrics or {}
        if self.metric_name in metrics:
            val = metrics[self.metric_name]
            is_best = (val < self.best_metric) if self.mode == "min" else (val > self.best_metric)
            if is_best:
                self.best_metric = val
                best_path = self.checkpoint_dir / "best.pt"
                save_checkpoint(
                    best_path, model, optimizer, step, tokens_seen, metrics, scheduler_state
                )
                logger.info(f"New best checkpoint at step {step}: {self.metric_name}={val:.4f}")

        # Rotate old checkpoints
        while len(self.saved_checkpoints) > self.max_to_keep:
            oldest = self.saved_checkpoints.pop(0)
            if oldest.exists() and oldest.name not in ("latest.pt", "best.pt"):
                oldest.unlink()
                meta = oldest.with_suffix(".meta.json")
                if meta.exists():
                    meta.unlink()

        # Update registry
        self._registry.append({
            "step": step,
            "tokens_seen": tokens_seen,
            "metrics": metrics,
            "file": ckpt_path.name,
            "saved_at": time.strftime("%Y-%m-%dT%H:%M:%SZ"),
        })
        self._save_registry()

        logger.info(f"Checkpoint saved: {ckpt_path.name} (step={step}, tokens={tokens_seen:,})")
        return ckpt_path

    # ── Load helpers ──────────────────────────────────────────

    def load_latest(
        self,
        model: Any,
        optimizer: Any = None,
        map_location: str = "cpu",
        restore_rng: bool = False,
    ) -> dict[str, Any]:
        """Load the latest checkpoint. Raises FileNotFoundError if missing."""
        latest_path = self.checkpoint_dir / "latest.pt"
        if not latest_path.exists():
            raise FileNotFoundError(
                f"No latest checkpoint found at {latest_path}. "
                "Did you mean to start a fresh training run (omit --resume)?"
            )
        state = load_checkpoint(latest_path, model, optimizer, map_location=map_location)
        if restore_rng and "rng_state" in state:
            _restore_rng_state(state["rng_state"])
        logger.info(f"Resumed from latest checkpoint: step={state.get('step')}, tokens={state.get('tokens_seen', 0):,}")
        return state

    def load_best(
        self,
        model: Any,
        optimizer: Any = None,
        map_location: str = "cpu",
    ) -> dict[str, Any]:
        """Load the best checkpoint."""
        best_path = self.checkpoint_dir / "best.pt"
        if not best_path.exists():
            raise FileNotFoundError(f"No best checkpoint found at {best_path}.")
        return load_checkpoint(best_path, model, optimizer, map_location=map_location)

    def verify_resume(self, expected_step: int, tolerance: int = 0) -> bool:
        """
        Verify that the latest checkpoint is at the expected step.
        Returns True if valid, False otherwise.
        """
        meta_path = self.checkpoint_dir / "latest.meta.json"
        if not meta_path.exists():
            return False
        try:
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
            actual_step = meta.get("step", -1)
            return abs(actual_step - expected_step) <= tolerance
        except Exception:
            return False

    def list_checkpoints(self) -> list[dict]:
        """Return a list of all saved checkpoint metadata from the registry."""
        return list(self._registry)

    def get_latest_meta(self) -> dict | None:
        """Return metadata of the latest checkpoint without loading tensors."""
        meta_path = self.checkpoint_dir / "latest.meta.json"
        if not meta_path.exists():
            return None
        try:
            return json.loads(meta_path.read_text(encoding="utf-8"))
        except Exception:
            return None
