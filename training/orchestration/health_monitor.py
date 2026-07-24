"""
Health monitoring for Vajra production training.
Monitors GPU, CPU, RAM, disk, loss spikes, gradient explosions, and OOM conditions.
"""

from __future__ import annotations

import math
import shutil
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Any

import torch

from utils.logging import setup_logger

logger = setup_logger("health_monitor")


@dataclass
class HealthSnapshot:
    timestamp: float = field(default_factory=time.time)
    gpu_util_pct: float = 0.0
    gpu_mem_allocated_mb: float = 0.0
    gpu_mem_reserved_mb: float = 0.0
    gpu_mem_total_mb: float = 0.0
    cpu_pct: float = 0.0
    ram_used_gb: float = 0.0
    ram_total_gb: float = 0.0
    disk_free_gb: float = 0.0
    disk_total_gb: float = 0.0
    tokens_per_sec: float = 0.0
    train_loss: float = 0.0
    grad_norm: float = 0.0
    warnings: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "timestamp": self.timestamp,
            "gpu_util_pct": round(self.gpu_util_pct, 1),
            "gpu_mem_allocated_mb": round(self.gpu_mem_allocated_mb, 1),
            "gpu_mem_reserved_mb": round(self.gpu_mem_reserved_mb, 1),
            "gpu_mem_total_mb": round(self.gpu_mem_total_mb, 1),
            "cpu_pct": round(self.cpu_pct, 1),
            "ram_used_gb": round(self.ram_used_gb, 2),
            "ram_total_gb": round(self.ram_total_gb, 2),
            "disk_free_gb": round(self.disk_free_gb, 2),
            "disk_total_gb": round(self.disk_total_gb, 2),
            "tokens_per_sec": round(self.tokens_per_sec),
            "train_loss": round(self.train_loss, 4),
            "grad_norm": round(self.grad_norm, 4),
            "warnings": self.warnings,
        }


class HealthMonitor:
    """
    Monitors system and training health, emitting warnings before failures.

    Checks:
    - GPU memory (warn at 90% utilization)
    - Disk space (warn below min_disk_free_gb)
    - Loss spikes (warn if loss jumps > spike_factor over moving average)
    - NaN/Inf loss
    - Gradient explosions
    - OOM signals
    """

    def __init__(
        self,
        checkpoint_dir: str = "checkpoints",
        min_disk_free_gb: float = 2.0,
        gpu_mem_warn_pct: float = 90.0,
        loss_history_len: int = 20,
        loss_spike_factor: float = 3.0,
        grad_norm_warn_threshold: float = 50.0,
    ) -> None:
        self.checkpoint_dir = checkpoint_dir
        self.min_disk_free_gb = min_disk_free_gb
        self.gpu_mem_warn_pct = gpu_mem_warn_pct
        self.loss_history: deque[float] = deque(maxlen=loss_history_len)
        self.loss_spike_factor = loss_spike_factor
        self.grad_norm_warn_threshold = grad_norm_warn_threshold

    def _gpu_stats(self) -> dict[str, float]:
        if not torch.cuda.is_available():
            return {}
        allocated = torch.cuda.memory_allocated() / (1024**2)
        reserved = torch.cuda.memory_reserved() / (1024**2)
        total = torch.cuda.get_device_properties(0).total_memory / (1024**2)
        return {
            "gpu_mem_allocated_mb": allocated,
            "gpu_mem_reserved_mb": reserved,
            "gpu_mem_total_mb": total,
        }

    def _sys_stats(self) -> dict[str, float]:
        stats: dict[str, float] = {}
        try:
            import psutil

            stats["cpu_pct"] = psutil.cpu_percent(interval=None)
            vm = psutil.virtual_memory()
            stats["ram_used_gb"] = vm.used / (1024**3)
            stats["ram_total_gb"] = vm.total / (1024**3)
        except ImportError:
            pass

        usage = shutil.disk_usage(self.checkpoint_dir)
        stats["disk_free_gb"] = usage.free / (1024**3)
        stats["disk_total_gb"] = usage.total / (1024**3)
        return stats

    def check(
        self,
        train_loss: float = 0.0,
        grad_norm: float = 0.0,
        tokens_per_sec: float = 0.0,
    ) -> HealthSnapshot:
        """Run a full health check and return a snapshot with any warnings."""
        snap = HealthSnapshot(
            train_loss=train_loss,
            grad_norm=grad_norm,
            tokens_per_sec=tokens_per_sec,
        )

        gpu = self._gpu_stats()
        snap.gpu_mem_allocated_mb = gpu.get("gpu_mem_allocated_mb", 0.0)
        snap.gpu_mem_reserved_mb = gpu.get("gpu_mem_reserved_mb", 0.0)
        snap.gpu_mem_total_mb = gpu.get("gpu_mem_total_mb", 0.0)

        sys = self._sys_stats()
        snap.cpu_pct = sys.get("cpu_pct", 0.0)
        snap.ram_used_gb = sys.get("ram_used_gb", 0.0)
        snap.ram_total_gb = sys.get("ram_total_gb", 0.0)
        snap.disk_free_gb = sys.get("disk_free_gb", 0.0)
        snap.disk_total_gb = sys.get("disk_total_gb", 0.0)

        # ── GPU memory warning ─────────────────────────────────
        if snap.gpu_mem_total_mb > 0:
            usage_pct = 100.0 * snap.gpu_mem_reserved_mb / snap.gpu_mem_total_mb
            if usage_pct >= self.gpu_mem_warn_pct:
                snap.warnings.append(
                    f"GPU memory at {usage_pct:.1f}% ({snap.gpu_mem_reserved_mb:.0f}/"
                    f"{snap.gpu_mem_total_mb:.0f} MB) — consider reducing batch size."
                )

        # ── Disk space warning ─────────────────────────────────
        if snap.disk_free_gb < self.min_disk_free_gb:
            snap.warnings.append(
                f"Low disk space: {snap.disk_free_gb:.2f} GB free "
                f"(threshold: {self.min_disk_free_gb} GB)."
            )

        # ── NaN / Inf loss ─────────────────────────────────────
        if math.isnan(train_loss) or math.isinf(train_loss):
            snap.warnings.append(f"Loss is NaN/Inf ({train_loss}) — training is diverging!")

        # ── Loss spike detection ───────────────────────────────
        if self.loss_history and not math.isnan(train_loss) and not math.isinf(train_loss):
            avg = sum(self.loss_history) / len(self.loss_history)
            if avg > 0 and train_loss > avg * self.loss_spike_factor:
                snap.warnings.append(
                    f"Loss spike detected: {train_loss:.4f} vs avg {avg:.4f} "
                    f"(factor {train_loss / avg:.1f}x)."
                )
        if not math.isnan(train_loss) and not math.isinf(train_loss):
            self.loss_history.append(train_loss)

        # ── Gradient explosion warning ─────────────────────────
        if grad_norm > self.grad_norm_warn_threshold:
            snap.warnings.append(
                f"High gradient norm: {grad_norm:.2f} (warn threshold: {self.grad_norm_warn_threshold})."
            )

        for w in snap.warnings:
            logger.warning(f"[HEALTH] {w}")

        return snap
