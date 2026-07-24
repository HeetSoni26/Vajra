"""
Training Orchestrator for Vajra — single entry point for the complete training lifecycle.

Coordinates:
  - Trainer          (optimization)
  - ResumeManager    (checkpoint discovery & restoration)
  - CloudSyncManager (background uploads & remote resume)
  - CheckpointManager (local save / rotation)
  - HealthMonitor    (GPU, CPU, disk, loss)
  - ExperimentManager (lifecycle, registry, summaries)
  - Watchdog         (freeze / deadlock detection)
  - ETAEngine        (ETA and progress)

Signal handling (SIGINT / SIGTERM) is installed to save an emergency checkpoint before exit.
"""

from __future__ import annotations

import signal
import time
from pathlib import Path
from typing import Any

import torch.distributed as dist
from torch.utils.data import DataLoader

from training.orchestration.experiment_manager import ExperimentManager, TrainingState
from training.orchestration.eta_engine import ETAEngine
from training.orchestration.health_monitor import HealthMonitor
from training.orchestration.watchdog import Watchdog
from training.cloud.sync_manager import CloudSyncManager
from training.trainer import Trainer
from training.training_logger import TrainingLogger
from utils.logging import setup_logger

logger = setup_logger("orchestrator")

_EMERGENCY_SAVE_REQUESTED = False


def _signal_handler(signum, _frame):  # noqa: ANN001
    global _EMERGENCY_SAVE_REQUESTED
    logger.warning(
        f"[SIGNAL] Received signal {signum}. Emergency checkpoint will be saved before exit."
    )
    _EMERGENCY_SAVE_REQUESTED = True


class TrainingOrchestrator:
    """
    Owns and coordinates the full production training lifecycle.
    Trainer is only responsible for gradient computation — everything else lives here.
    """

    def __init__(
        self,
        trainer: Trainer,
        train_loader: DataLoader,
        val_loader: DataLoader | None,
        exp_dir: Path,
        cfg: dict[str, Any],
        rank: int = 0,
        world_size: int = 1,
        is_distributed: bool = False,
        training_logger: TrainingLogger | None = None,
        resume_state: dict[str, Any] | None = None,
    ) -> None:
        self.trainer = trainer
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.exp_dir = exp_dir
        self.cfg = cfg
        self.rank = rank
        self.world_size = world_size
        self.is_distributed = is_distributed
        self.training_logger = training_logger

        self.max_steps = int(cfg.get("max_steps", 50))
        self.save_every = int(cfg.get("save_every_steps", 10))
        self.eval_every = int(cfg.get("eval_every_steps", 10))
        self.time_save_every_minutes = float(cfg.get("time_checkpoint_every_minutes", 15.0))

        # Sub-managers
        self.exp_manager = ExperimentManager(exp_dir)
        self.health_monitor = HealthMonitor(checkpoint_dir=str(exp_dir))
        self.eta_engine = ETAEngine(total_steps=self.max_steps)
        self.cloud_sync = CloudSyncManager()
        self._watchdog: Watchdog | None = None

        # Resume state
        self.start_step = 0
        self.tokens_seen = 0
        if resume_state is not None:
            self.start_step = resume_state.get("step", 0)
            self.tokens_seen = resume_state.get("tokens_seen", 0)

        # Install signal handlers on rank 0 only
        if rank == 0:
            signal.signal(signal.SIGINT, _signal_handler)
            signal.signal(signal.SIGTERM, _signal_handler)

    def _start_watchdog(self) -> None:
        timeout = float(self.cfg.get("watchdog_timeout_seconds", 300.0))

        def _emergency_save():
            logger.error("[WATCHDOG] Emergency checkpoint save triggered.")
            if self.rank == 0:
                self.trainer.checkpoint_manager.save(
                    step=self._global_step,
                    model=self.trainer.raw_model,
                    optimizer=self.trainer.optimizer,
                    tokens_seen=self.tokens_seen,
                    metrics={"emergency": True},
                )
                self.cloud_sync.sync_experiment(self.exp_dir)

        self._watchdog = Watchdog(
            timeout_seconds=timeout,
            on_trigger=_emergency_save,
        )
        self._watchdog.start()

    def run(self) -> dict[str, Any]:
        """Execute the complete training loop, managing lifecycle transitions."""
        global _EMERGENCY_SAVE_REQUESTED

        self.exp_manager.transition(TrainingState.TRAINING)
        self.exp_manager.record_provider()
        self._start_watchdog()

        train_iter = iter(self.train_loader)
        history: list[dict[str, Any]] = []
        self._global_step = self.start_step
        micro_step = 0
        grad_accum = max(1, int(self.cfg.get("gradient_accumulation_steps", 1)))

        last_time_ckpt = time.time()

        if self.rank == 0:
            logger.info(
                f"[ORCHESTRATOR] Training started | steps={self.max_steps} "
                f"| start_step={self.start_step} | tokens_seen={self.tokens_seen:,}"
            )

        try:
            while self._global_step < self.max_steps:
                # ── Emergency exit check ───────────────────────
                if _EMERGENCY_SAVE_REQUESTED:
                    logger.warning("[ORCHESTRATOR] Interrupted — saving emergency checkpoint.")
                    if self.rank == 0:
                        self._save_checkpoint(emergency=True)
                    self.exp_manager.transition(TrainingState.INTERRUPTED)
                    break

                # ── Fetch batch ────────────────────────────────
                try:
                    batch = next(train_iter)
                except StopIteration:
                    train_iter = iter(self.train_loader)
                    batch = next(train_iter)

                micro_step += 1
                is_boundary = micro_step % grad_accum == 0
                step_metrics = self.trainer.train_step(
                    batch, step=self._global_step, is_accum_step=is_boundary
                )

                if not is_boundary or step_metrics is None:
                    continue

                self._global_step += 1
                self.tokens_seen += step_metrics.get("tokens_processed", 0) * self.world_size

                # Watchdog heartbeat
                if self._watchdog:
                    self._watchdog.heartbeat()

                # ETA
                self.eta_engine.record_step()
                progress = self.eta_engine.get_progress(self._global_step, self.tokens_seen)

                # ── Validation ─────────────────────────────────
                if self.val_loader and self._global_step % self.eval_every == 0:
                    self.exp_manager.transition(TrainingState.VALIDATING)
                    val_stats = self.trainer.evaluate(self.val_loader)
                    step_metrics.update(val_stats)
                    self.exp_manager.transition(TrainingState.TRAINING)

                # ── Health check ───────────────────────────────
                health = self.health_monitor.check(
                    train_loss=step_metrics.get("loss", 0.0),
                    grad_norm=step_metrics.get("grad_norm", 0.0),
                    tokens_per_sec=step_metrics.get("tokens_per_sec", 0.0),
                )

                # ── Step / time-based checkpointing ───────────
                time_since_ckpt = (time.time() - last_time_ckpt) / 60.0
                should_step_ckpt = (
                    self._global_step % self.save_every == 0 or self._global_step == self.max_steps
                )
                should_time_ckpt = time_since_ckpt >= self.time_save_every_minutes

                if self.rank == 0 and (should_step_ckpt or should_time_ckpt):
                    self._save_checkpoint(step_metrics)
                    last_time_ckpt = time.time()

                if self.is_distributed and dist.is_initialized():
                    dist.barrier()

                # ── Logging ────────────────────────────────────
                if self.rank == 0:
                    step_metrics["progress"] = progress
                    step_metrics["health"] = health.to_dict()
                    history.append(step_metrics)
                    if self.training_logger:
                        self.training_logger.log_step(self._global_step, step_metrics)
                    if self._global_step % max(1, self.save_every // 5) == 0:
                        logger.info(
                            f"Step {self._global_step:5d}/{self.max_steps} | "
                            f"Loss: {step_metrics['loss']:.4f} | "
                            f"LR: {step_metrics['learning_rate']:.2e} | "
                            f"Tok/s: {progress['tokens_per_sec']:,} | "
                            f"ETA: {progress['eta_hours']:.2f}h | "
                            f"{progress['pct_complete']:.1f}% complete"
                        )

            else:
                # Normal completion
                self.exp_manager.transition(TrainingState.COMPLETED)

        except Exception as exc:
            logger.exception(f"[ORCHESTRATOR] Training failed with exception: {exc}")
            self.exp_manager.transition(TrainingState.FAILED)
            if self.rank == 0:
                self._save_checkpoint(emergency=True)
            raise

        finally:
            if self._watchdog:
                self._watchdog.stop()

        # ── Final summary ──────────────────────────────────────
        summary = {
            "completed_steps": self._global_step,
            "total_steps": self.max_steps,
            "tokens_seen": self.tokens_seen,
            "final_loss": history[-1]["loss"] if history else None,
            "final_val_loss": history[-1].get("val_loss") if history else None,
        }
        self.exp_manager.record_summary(summary)
        return summary

    def _save_checkpoint(
        self,
        step_metrics: dict[str, Any] | None = None,
        emergency: bool = False,
    ) -> None:
        """Save checkpoint and trigger background cloud sync."""
        self.exp_manager.transition(TrainingState.CHECKPOINTING)
        metrics = (
            {"val_loss": step_metrics.get("val_loss", step_metrics.get("loss", 0.0))}
            if step_metrics
            else {}
        )
        if emergency:
            metrics["emergency"] = True

        ckpt_path = self.trainer.checkpoint_manager.save(
            step=self._global_step,
            model=self.trainer.raw_model,
            optimizer=self.trainer.optimizer,
            tokens_seen=self.tokens_seen,
            metrics=metrics,
        )
        self.exp_manager.record_checkpoint(
            step=self._global_step,
            tokens_seen=self.tokens_seen,
            path=str(ckpt_path),
            metrics=metrics,
        )

        if self.training_logger:
            self.training_logger.log_checkpoint(self._global_step, ckpt_path.name)

        # ── Background cloud upload ─────────────────────────────
        self.exp_manager.transition(TrainingState.UPLOADING)
        self.cloud_sync.sync_experiment(self.exp_dir)
        self.exp_manager.transition(TrainingState.TRAINING)
