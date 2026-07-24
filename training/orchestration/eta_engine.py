"""
ETA and progress engine for Vajra training.
"""
from __future__ import annotations

import time


class ETAEngine:
    """Calculates training ETA from throughput history."""

    def __init__(self, total_steps: int) -> None:
        self.total_steps = total_steps
        self._start_time = time.time()
        self._step_times: list[float] = []
        self._last_step_time = time.time()

    def record_step(self) -> None:
        now = time.time()
        self._step_times.append(now - self._last_step_time)
        self._last_step_time = now
        if len(self._step_times) > 100:
            self._step_times.pop(0)

    def get_progress(self, current_step: int, tokens_seen: int) -> dict[str, object]:
        elapsed = time.time() - self._start_time
        remaining_steps = max(0, self.total_steps - current_step)

        avg_step_time = (sum(self._step_times) / len(self._step_times)) if self._step_times else 0.0
        eta_seconds = avg_step_time * remaining_steps if avg_step_time > 0 else 0.0

        tokens_per_sec = tokens_seen / elapsed if elapsed > 0 else 0.0

        return {
            "current_step": current_step,
            "total_steps": self.total_steps,
            "remaining_steps": remaining_steps,
            "elapsed_s": round(elapsed, 1),
            "eta_s": round(eta_seconds, 1),
            "eta_hours": round(eta_seconds / 3600, 2),
            "tokens_seen": tokens_seen,
            "tokens_per_sec": round(tokens_per_sec),
            "pct_complete": round(100.0 * current_step / max(1, self.total_steps), 1),
        }
