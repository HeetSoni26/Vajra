import math
import torch
from torch.optim.lr_scheduler import LambdaLR
from training.config import TrainingConfig


def create_scheduler(
    optimizer: torch.optim.Optimizer, config: TrainingConfig
) -> torch.optim.lr_scheduler.LRScheduler:
    """
    Creates the learning rate scheduler based on the TrainingConfig.
    """
    warmup_steps = config.warmup_steps
    max_steps = config.max_steps

    if config.lr_scheduler_type == "constant":

        def lr_lambda(current_step: int):
            if current_step < warmup_steps:
                return float(current_step) / float(max(1, warmup_steps))
            return 1.0

    elif config.lr_scheduler_type == "linear":

        def lr_lambda(current_step: int):
            if current_step < warmup_steps:
                return float(current_step) / float(max(1, warmup_steps))
            return max(
                0.0, float(max_steps - current_step) / float(max(1, max_steps - warmup_steps))
            )

    elif config.lr_scheduler_type == "cosine":

        def lr_lambda(current_step: int):
            if current_step < warmup_steps:
                return float(current_step) / float(max(1, warmup_steps))
            progress = float(current_step - warmup_steps) / float(max(1, max_steps - warmup_steps))
            return max(0.0, 0.5 * (1.0 + math.cos(math.pi * progress)))

    elif config.lr_scheduler_type == "step":
        # Hardcoded to drop by 0.1 every half of remaining steps
        def lr_lambda(current_step: int):
            if current_step < warmup_steps:
                return float(current_step) / float(max(1, warmup_steps))
            if current_step < max_steps // 2:
                return 1.0
            if current_step < max_steps // 4 * 3:
                return 0.1
            return 0.01
    else:
        raise ValueError(f"Unknown scheduler type {config.lr_scheduler_type}")

    return LambdaLR(optimizer, lr_lambda)
