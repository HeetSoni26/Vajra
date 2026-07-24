import torch
from torch import nn
from torch.nn.parallel import DistributedDataParallel as DDP

from model.modeling import VajraForCausalLM
from training.ddp.config import DDPConfig


def wrap_model_ddp(model: VajraForCausalLM, config: DDPConfig, device: torch.device) -> DDP:
    """
    Places the model on the given device and wraps it in DistributedDataParallel.
    """
    model = model.to(device)
    ddp_model = DDP(
        model,
        device_ids=[device.index] if device.type == "cuda" else None,
        output_device=device.index if device.type == "cuda" else None,
        find_unused_parameters=config.find_unused_parameters,
        broadcast_buffers=config.broadcast_buffers,
        static_graph=config.static_graph,
    )
    return ddp_model


def unwrap_model(model: nn.Module) -> nn.Module:
    """
    Unwraps a DDP-wrapped model to get the underlying module (e.g., for checkpointing).
    """
    if isinstance(model, DDP):
        return model.module
    return model
