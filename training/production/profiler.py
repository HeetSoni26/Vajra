import time

import torch


class MemoryProfiler:
    """Tracks and reports memory utilization during training."""

    def __init__(self, device: torch.device):
        self.device = device
        self.enabled = device.type == "cuda"

    def get_memory_stats(self) -> dict[str, float]:
        if not self.enabled:
            return {}

        stats = {
            "memory_allocated_mb": torch.cuda.memory_allocated(self.device) / (1024**2),
            "memory_reserved_mb": torch.cuda.memory_reserved(self.device) / (1024**2),
            "max_memory_allocated_mb": torch.cuda.max_memory_allocated(self.device) / (1024**2),
        }
        return stats

    def reset_peak_stats(self):
        if self.enabled:
            torch.cuda.reset_peak_memory_stats(self.device)


class PerformanceProfiler:
    """Tracks latency and throughput during training."""

    def __init__(self):
        self.step_start_time = 0.0
        self.forward_start_time = 0.0
        self.backward_start_time = 0.0

        self.stats = {
            "forward_time_ms": 0.0,
            "backward_time_ms": 0.0,
            "step_time_ms": 0.0,
        }

    def start_step(self):
        self.step_start_time = time.perf_counter()

    def start_forward(self):
        self.forward_start_time = time.perf_counter()

    def end_forward(self):
        self.stats["forward_time_ms"] = (time.perf_counter() - self.forward_start_time) * 1000

    def start_backward(self):
        self.backward_start_time = time.perf_counter()

    def end_backward(self):
        self.stats["backward_time_ms"] = (time.perf_counter() - self.backward_start_time) * 1000

    def end_step(self) -> dict[str, float]:
        self.stats["step_time_ms"] = (time.perf_counter() - self.step_start_time) * 1000
        return self.stats.copy()
