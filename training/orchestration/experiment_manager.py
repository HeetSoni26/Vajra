"""
Experiment lifecycle management for Vajra.
Tracks experiment metadata, checkpoint history, provider history, and training summaries.
"""

from __future__ import annotations

import json
import os
import socket
import time
from enum import Enum
from pathlib import Path
from typing import Any

import torch

from utils.logging import setup_logger

logger = setup_logger("experiment_manager")


class TrainingState(str, Enum):
    INITIALIZING = "INITIALIZING"
    RESTORING = "RESTORING"
    TRAINING = "TRAINING"
    VALIDATING = "VALIDATING"
    CHECKPOINTING = "CHECKPOINTING"
    UPLOADING = "UPLOADING"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"
    INTERRUPTED = "INTERRUPTED"


class ExperimentManager:
    """
    Manages the full lifecycle of a training experiment.

    Maintains a JSON-backed registry of all experiments including:
    - state transitions with timestamps
    - checkpoint history
    - provider / hardware information
    - resume history
    - runtime statistics
    """

    REGISTRY_FILENAME = "experiment_registry.json"

    def __init__(self, exp_dir: str | Path) -> None:
        self.exp_dir = Path(exp_dir)
        self.exp_dir.mkdir(parents=True, exist_ok=True)
        self.registry_path = self.exp_dir / self.REGISTRY_FILENAME

        self._record: dict[str, Any] = self._load_or_init()
        self.state = TrainingState(self._record.get("state", TrainingState.INITIALIZING))
        # Persist initial state immediately so the registry file always exists
        self._save()

    # ── Persistence ────────────────────────────────────────────

    def _load_or_init(self) -> dict[str, Any]:
        if self.registry_path.exists():
            try:
                return json.loads(self.registry_path.read_text(encoding="utf-8"))
            except Exception:
                pass
        return {
            "experiment_id": self.exp_dir.name,
            "state": TrainingState.INITIALIZING,
            "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ"),
            "state_history": [],
            "checkpoint_history": [],
            "resume_history": [],
            "provider_history": [],
            "runtime_stats": {},
            "training_summary": {},
        }

    def _save(self) -> None:
        self._record["state"] = self.state.value
        self._record["updated_at"] = time.strftime("%Y-%m-%dT%H:%M:%SZ")
        self.registry_path.write_text(json.dumps(self._record, indent=2), encoding="utf-8")

    # ── State Transitions ──────────────────────────────────────

    def transition(self, new_state: TrainingState) -> None:
        old = self.state
        self.state = new_state
        entry = {
            "from": old.value,
            "to": new_state.value,
            "at": time.strftime("%Y-%m-%dT%H:%M:%SZ"),
        }
        self._record["state_history"].append(entry)
        logger.info(f"[LIFECYCLE] {old.value} → {new_state.value}")
        self._save()

    # ── Checkpoint Registry ────────────────────────────────────

    def record_checkpoint(self, step: int, tokens_seen: int, path: str, metrics: dict) -> None:
        self._record["checkpoint_history"].append(
            {
                "step": step,
                "tokens_seen": tokens_seen,
                "path": str(path),
                "metrics": metrics,
                "saved_at": time.strftime("%Y-%m-%dT%H:%M:%SZ"),
            }
        )
        self._save()

    # ── Resume Registry ────────────────────────────────────────

    def record_resume(self, from_step: int, from_exp: str) -> None:
        self._record["resume_history"].append(
            {
                "from_step": from_step,
                "from_experiment": from_exp,
                "resumed_at": time.strftime("%Y-%m-%dT%H:%M:%SZ"),
            }
        )
        self._save()

    # ── Provider Info ──────────────────────────────────────────

    def record_provider(self) -> None:
        entry: dict[str, Any] = {
            "hostname": socket.gethostname(),
            "recorded_at": time.strftime("%Y-%m-%dT%H:%M:%SZ"),
        }
        if torch.cuda.is_available():
            entry["gpu"] = torch.cuda.get_device_name(0)
            entry["gpu_memory_total_gb"] = round(
                torch.cuda.get_device_properties(0).total_memory / (1024**3), 2
            )
        entry["env_vars"] = {
            k: os.environ.get(k, "")
            for k in ("KAGGLE_KERNEL_RUN_TYPE", "COLAB_BACKEND_VERSION", "RUNPOD_POD_ID")
            if os.environ.get(k)
        }
        self._record["provider_history"].append(entry)
        self._save()

    # ── Summary ───────────────────────────────────────────────

    def record_summary(self, stats: dict[str, Any]) -> None:
        self._record["training_summary"] = stats
        self._save()

    def update_runtime_stats(self, **kwargs: Any) -> None:
        self._record["runtime_stats"].update(kwargs)
        self._save()

    # ── Query ─────────────────────────────────────────────────

    def get_record(self) -> dict[str, Any]:
        return dict(self._record)
