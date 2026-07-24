"""
Resume management for Vajra production training.
Provides a robust ResumeManager to discover, validate, and restore experiments seamlessly.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
from torch import nn

from training.checkpoint import _restore_rng_state
from training.cloud.sync_manager import CloudSyncManager
from utils.logging import setup_logger

logger = setup_logger("resume_manager")


class CheckpointValidationError(Exception):
    """Raised when a checkpoint fails validation."""


class ResumeManager:
    """
    Orchestrates discovering, validating, and loading training checkpoints
    to ensure training can survive interruptions seamlessly.
    """

    def __init__(self, base_dir: str | Path) -> None:
        self.base_dir = Path(base_dir)
        self.cloud_sync = CloudSyncManager()

    def discover_experiments(self, prefix: str = "exp_") -> list[Path]:
        """
        Scan base_dir for experiment directories, sorted by newest first.
        Expects directories named with a timestamp, e.g., 'exp_YYYYMMDD_HHMMSS'.
        """
        if not self.base_dir.exists():
            return []

        exps = [d for d in self.base_dir.iterdir() if d.is_dir() and d.name.startswith(prefix)]
        # Sort descending (newest first) based on directory name
        exps.sort(key=lambda p: p.name, reverse=True)
        return exps

    def validate_checkpoint(self, checkpoint_path: Path) -> dict[str, Any]:
        """
        Validate that a checkpoint file is intact, readable, and contains the required keys.
        Returns the loaded state dict if valid.
        Raises CheckpointValidationError if corrupted or invalid.
        """
        if not checkpoint_path.exists():
            raise CheckpointValidationError(f"Checkpoint file not found: {checkpoint_path}")

        try:
            # We load to CPU for validation to avoid OOM
            state = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        except Exception as e:
            raise CheckpointValidationError(f"Failed to read checkpoint {checkpoint_path}: {e}")

        # Verify required top-level keys
        required_keys = ["model", "optimizer", "scheduler", "step", "tokens_seen", "rng_state"]
        missing = [k for k in required_keys if k not in state]
        if missing:
            raise CheckpointValidationError(f"Checkpoint {checkpoint_path} missing keys: {missing}")

        return state

    def find_latest_valid_experiment(self, prefix: str = "exp_") -> tuple[Path, dict[str, Any]]:
        """
        Search previous experiments, find the latest one with a valid `latest.pt`.
        Returns (experiment_dir, state_dict).
        Raises FileNotFoundError if no valid experiment could be found.
        """
        experiments = self.discover_experiments(prefix=prefix)
        logger.info("=" * 48)
        logger.info(f"Found {len(experiments)} local experiments in {self.base_dir}")

        for exp_dir in experiments:
            ckpt_path = exp_dir / "latest.pt"
            if not ckpt_path.exists():
                logger.debug(f"Skipping {exp_dir.name} (no latest.pt)")
                continue

            try:
                state = self.validate_checkpoint(ckpt_path)
                logger.info(f"Latest valid experiment: {exp_dir.name}")
                logger.info(f"Checkpoint: {ckpt_path.name}")
                logger.info(f"Global step: {state['step']}")
                logger.info(f"Tokens seen: {state['tokens_seen']:,}")
                logger.info("Checkpoint verified.")
                logger.info("=" * 48)
                return exp_dir, state
            except CheckpointValidationError as e:
                logger.warning(f"Skipping {exp_dir.name} (corrupted checkpoint): {e}")

        # If we failed to find locally, try remote
        if self.cloud_sync.enabled and self.cloud_sync.config.get("download_on_resume", True):
            logger.info("No valid local experiment found. Querying cloud provider...")
            remote_exps = self.cloud_sync.discover_remote_experiments()
            if remote_exps:
                # We try the most recent remote experiment
                latest_remote = remote_exps[0]
                local_dir = self.cloud_sync.download_experiment(latest_remote, self.base_dir)
                if local_dir:
                    ckpt_path = local_dir / "latest.pt"
                    try:
                        state = self.validate_checkpoint(ckpt_path)
                        logger.info(f"Successfully recovered {latest_remote} from cloud.")
                        return local_dir, state
                    except CheckpointValidationError as e:
                        logger.error(f"Downloaded remote checkpoint is corrupted: {e}")

        raise FileNotFoundError(
            f"No valid resume state found locally or remotely in {self.base_dir}"
        )

    def restore_state(
        self,
        state: dict[str, Any],
        model: nn.Module,
        optimizer: torch.optim.Optimizer | None = None,
        scaler: torch.amp.GradScaler | None = None,
        device: torch.device | str = "cpu",
    ) -> tuple[int, int]:
        """
        Restore model, optimizer, scaler, and RNG states.
        Returns (global_step, tokens_seen).
        """
        logger.info("Restoring training state...")

        # 1. Restore Model
        # Load state dict directly to model
        if hasattr(model, "module"):  # DDP model
            model.module.load_state_dict(state["model"])
        else:
            model.load_state_dict(state["model"])

        # Move model to device just in case
        model.to(device)

        # 2. Restore Optimizer
        if optimizer is not None and state.get("optimizer") is not None:
            optimizer.load_state_dict(state["optimizer"])

        # 3. Restore Scaler (if applicable and saved)
        # Assuming scaler might not be in state if not explicitly saved by CheckpointManager previously,
        # but if we add it we can restore it. Currently CheckpointManager doesn't save scaler state,
        # so we'll skip scaler for now to maintain backward compatibility, or add support if it exists.
        if scaler is not None and "scaler" in state and state["scaler"] is not None:
            scaler.load_state_dict(state["scaler"])

        # 4. Restore RNG
        if "rng_state" in state:
            _restore_rng_state(state["rng_state"])

        logger.info("Resume successful.")

        return state.get("step", 0), state.get("tokens_seen", 0)
