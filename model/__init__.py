from model.checkpoints import CheckpointManager
from model.config import ModelConfig, VajraConfig, get_preset
from model.generation.engine import GenerationEngine
from model.modeling import VajraForCausalLM, VajraModel
from model.utils import count_parameters, summarize_model

# Canonical Aliases for Backward Compatibility
FoundationLM = VajraForCausalLM

__all__ = [
    "CheckpointManager",
    "FoundationLM",
    "GenerationEngine",
    "ModelConfig",
    "VajraConfig",
    "VajraForCausalLM",
    "VajraModel",
    "count_parameters",
    "get_preset",
    "summarize_model",
]
