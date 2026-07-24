import math
import torch

from training.metrics.tracker import MetricsTracker


def perplexity(loss: float) -> float:
    """Compute perplexity given loss value."""
    return math.exp(min(20.0, float(loss)))


def get_gpu_memory_mb() -> float:
    """Get peak allocated CUDA VRAM memory in MB if available."""
    if torch.cuda.is_available():
        return round(torch.cuda.max_memory_allocated() / (1024**2), 2)
    return 0.0


__all__ = ["MetricsTracker", "perplexity", "get_gpu_memory_mb"]
