from collections.abc import Iterable
from typing import Any

import torch
from torch.optim import AdamW

from training.config import TrainingConfig
from training.production.config import OptimisationConfig


def create_production_optimizer(
    model_parameters: Iterable[Any],
    training_config: TrainingConfig,
    optim_config: OptimisationConfig,
) -> torch.optim.Optimizer:
    """
    Creates an optimized optimizer backend (e.g. Fused AdamW).
    """
    # Try fused AdamW first if requested
    use_fused = optim_config.fused_optimizer and torch.cuda.is_available()

    try:
        optimizer = AdamW(
            model_parameters,
            lr=training_config.learning_rate,
            weight_decay=training_config.weight_decay,
            betas=(0.9, 0.95),
            fused=use_fused,
        )
    except TypeError:
        # Fallback if fused is not supported in this torch version
        optimizer = AdamW(
            model_parameters,
            lr=training_config.learning_rate,
            weight_decay=training_config.weight_decay,
            betas=(0.9, 0.95),
        )

    return optimizer
