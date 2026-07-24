import torch
import torch.nn as nn
from training.config import TrainingConfig


def create_optimizer(model: nn.Module, config: TrainingConfig) -> torch.optim.Optimizer:
    """
    Creates an AdamW optimizer adhering to standard LLM training practices.
    Separates biases and LayerNorm weights from weight decay.
    """
    decay_params = []
    no_decay_params = []

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue

        if param.dim() < 2 or "norm" in name or "bias" in name:
            no_decay_params.append(param)
        else:
            decay_params.append(param)

    optim_groups = [
        {"params": decay_params, "weight_decay": config.weight_decay},
        {"params": no_decay_params, "weight_decay": 0.0},
    ]

    return torch.optim.AdamW(
        optim_groups,
        lr=config.learning_rate,
        betas=(config.adam_beta1, config.adam_beta2),
        eps=config.adam_eps,
    )
