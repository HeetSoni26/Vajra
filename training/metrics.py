from __future__ import annotations

import math
import time
from typing import Any

import torch


def perplexity(loss: float) -> float:
    """Compute perplexity given loss value."""
    return math.exp(min(20.0, float(loss)))


def get_gpu_memory_mb() -> float:
    """Get peak allocated CUDA VRAM memory in MB if available."""
    if torch.cuda.is_available():
        return round(torch.cuda.max_memory_allocated() / (1024**2), 2)
    return 0.0


class MetricsTracker:
    """Tracks training statistics, throughput, and system resource metrics."""

    def __init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        self.history: list[dict[str, Any]] = []
        self.step_start_time = time.time()
        self.tokens_seen_total = 0

    def record_step(
        self,
        step: int,
        loss: float,
        lr: float,
        tokens_in_step: int,
        samples_in_step: int,
        grad_norm: float = 0.0,
        val_loss: float | None = None,
    ) -> dict[str, Any]:
        """Record metrics for a single training step."""
        now = time.time()
        elapsed = max(0.0001, now - self.step_start_time)
        self.step_start_time = now

        self.tokens_seen_total += tokens_in_step
        tokens_sec = round(tokens_in_step / elapsed, 2)
        samples_sec = round(samples_in_step / elapsed, 2)
        gpu_mem_mb = get_gpu_memory_mb()

        entry = {
            "step": step,
            "train_loss": round(float(loss), 4),
            "val_loss": round(float(val_loss), 4) if val_loss is not None else None,
            "perplexity": round(perplexity(loss), 4),
            "learning_rate": lr,
            "tokens_per_sec": tokens_sec,
            "samples_per_sec": samples_sec,
            "grad_norm": round(float(grad_norm), 4),
            "gpu_memory_mb": gpu_mem_mb,
            "tokens_processed": self.tokens_seen_total,
        }
        self.history.append(entry)
        return entry
