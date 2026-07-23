"""
Production logging infrastructure for Vajra training.

Supports:
- Structured JSON log (training_log.jsonl)
- CSV log (training_log.csv)
- TensorBoard (optional, graceful fallback)
- Weights & Biases (optional, graceful fallback)
- Human-readable console output
"""

from __future__ import annotations

import csv
import json
import os
import time
from pathlib import Path
from typing import Any

from utils.logging import setup_logger

logger = setup_logger("training_logger")


class TrainingLogger:
    """
    Unified training logger supporting CSV, JSONL, TensorBoard, and W&B.
    All external logging backends degrade gracefully if unavailable.
    """

    def __init__(
        self,
        log_dir: str | Path,
        run_name: str = "vajra-run",
        use_tensorboard: bool = True,
        use_wandb: bool = True,
        wandb_project: str | None = None,
        wandb_config: dict | None = None,
        rank: int = 0,
    ) -> None:
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.run_name = run_name
        self.rank = rank  # Only rank 0 writes logs

        self._csv_path = self.log_dir / "training_log.csv"
        self._jsonl_path = self.log_dir / "training_log.jsonl"
        self._csv_writer: csv.DictWriter | None = None
        self._csv_file = None
        self._fieldnames: list[str] | None = None

        self._tb_writer = None
        self._wandb_run = None

        if rank != 0:
            return  # Only primary rank initializes backends

        # TensorBoard
        if use_tensorboard:
            try:
                from torch.utils.tensorboard import SummaryWriter
                tb_dir = self.log_dir / "tensorboard"
                self._tb_writer = SummaryWriter(log_dir=str(tb_dir))
                logger.info(f"TensorBoard writer initialized at {tb_dir}")
            except Exception as e:
                logger.warning(f"TensorBoard unavailable (will skip): {e}")

        # Weights & Biases
        if use_wandb and wandb_project:
            wandb_api_key = os.environ.get("WANDB_API_KEY", "")
            if wandb_api_key:
                try:
                    import wandb
                    self._wandb_run = wandb.init(
                        project=wandb_project,
                        name=run_name,
                        config=wandb_config or {},
                        resume="allow",
                    )
                    logger.info(f"W&B run initialized: project={wandb_project}, run={run_name}")
                except Exception as e:
                    logger.warning(f"W&B initialization failed (will skip): {e}")
            else:
                logger.info("WANDB_API_KEY not set — W&B logging disabled (training continues normally)")

    def log_step(self, step: int, metrics: dict[str, Any]) -> None:
        """Log a training step to all available backends."""
        if self.rank != 0:
            return

        record = {"step": step, "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ"), **metrics}

        # JSONL
        with open(self._jsonl_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(record) + "\n")

        # CSV
        self._write_csv(record)

        # TensorBoard
        if self._tb_writer is not None:
            try:
                for k, v in metrics.items():
                    if isinstance(v, (int, float)):
                        self._tb_writer.add_scalar(k, v, global_step=step)
            except Exception:
                pass

        # W&B
        if self._wandb_run is not None:
            try:
                self._wandb_run.log({**metrics, "step": step})
            except Exception:
                pass

    def log_checkpoint(self, step: int, ckpt_path: str) -> None:
        """Record a checkpoint event in the log."""
        if self.rank != 0:
            return
        record = {
            "event": "checkpoint",
            "step": step,
            "checkpoint": ckpt_path,
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ"),
        }
        with open(self._jsonl_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(record) + "\n")

    def log_summary(self, summary: dict[str, Any]) -> None:
        """Write a run summary."""
        if self.rank != 0:
            return
        summary_path = self.log_dir / "run_summary.json"
        summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
        logger.info(f"Run summary written to {summary_path}")

    def _write_csv(self, record: dict[str, Any]) -> None:
        """Lazily initialise the CSV writer and append the row."""
        flat = {k: v for k, v in record.items() if isinstance(v, (str, int, float, bool))}
        if self._fieldnames is None:
            self._fieldnames = list(flat.keys())
            self._csv_file = open(self._csv_path, "w", newline="", encoding="utf-8")
            self._csv_writer = csv.DictWriter(self._csv_file, fieldnames=self._fieldnames, extrasaction="ignore")
            self._csv_writer.writeheader()

        # Extend fieldnames if new keys appear
        for k in flat:
            if k not in self._fieldnames:
                self._fieldnames.append(k)

        self._csv_writer.writerow(flat)
        self._csv_file.flush()

    def close(self) -> None:
        """Flush and close all logging backends."""
        if self._tb_writer is not None:
            try:
                self._tb_writer.close()
            except Exception:
                pass
        if self._wandb_run is not None:
            try:
                self._wandb_run.finish()
            except Exception:
                pass
        if self._csv_file is not None:
            try:
                self._csv_file.close()
            except Exception:
                pass
