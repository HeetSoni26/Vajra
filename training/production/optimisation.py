import torch
from torch import nn
from torch.utils.checkpoint import checkpoint

from training.production.config import OptimisationConfig


def apply_gradient_checkpointing(model: nn.Module) -> nn.Module:
    """
    Applies gradient checkpointing to the transformer layers of the model.
    Assumes the model exposes its transformer layers as `model.layers` or similar.
    """
    if hasattr(model, "layers"):
        layers = model.layers
    elif hasattr(model, "transformer") and hasattr(model.transformer, "h"):
        layers = model.transformer.h
    elif hasattr(model, "model") and hasattr(model.model, "layers"):
        layers = model.model.layers
    else:
        # Fallback: cannot auto-detect layers
        return model

    for layer in layers:
        # Save original forward pass
        original_forward = layer.forward

        # Create a checkpointed forward pass
        def checkpointed_forward(*args, original_forward=original_forward, **kwargs):
            # torch.utils.checkpoint.checkpoint requires inputs to require_grad.
            # Usually kwargs are not easily passed through old checkpoint APIs,
            # but in recent PyTorch versions kwargs are supported via use_reentrant=False.
            return checkpoint(original_forward, *args, use_reentrant=False, **kwargs)

        layer.forward = checkpointed_forward

    return model


def apply_compilation(model: nn.Module, config: OptimisationConfig) -> nn.Module:
    """
    Applies torch.compile if supported and enabled.
    """
    if not config.compile_model:
        return model

    if not hasattr(torch, "compile"):
        print("[WARNING] torch.compile is not available in this PyTorch version.")
        return model

    # In a real scenario we'd do dynamic checks, but for now we trust the config.
    try:
        compiled_model = torch.compile(model, backend=config.compile_backend)
        return compiled_model
    except Exception as e:
        print(f"[WARNING] torch.compile failed: {e}. Falling back to eager execution.")
        return model


def optimise_model_for_production(model: nn.Module, config: OptimisationConfig) -> nn.Module:
    """
    Applies production optimisations to the model:
    1. Gradient Checkpointing
    2. torch.compile
    """
    if config.gradient_checkpointing:
        model = apply_gradient_checkpointing(model)

    if config.compile_model:
        model = apply_compilation(model, config)

    return model
