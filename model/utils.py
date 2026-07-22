import torch.nn as nn
from typing import Dict, Any
from model.modeling import VajraForCausalLM
from model.config import VajraConfig

def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def summarize_model(model: VajraForCausalLM) -> Dict[str, Any]:
    return {
        "model_type": "VajraForCausalLM",
        "parameters": count_parameters(model),
        "layers": model.config.num_layers,
        "vocab_size": model.config.vocab_size,
        "hidden_size": model.config.hidden_size,
        "attention_heads": model.config.num_attention_heads,
        "kv_heads": model.config.num_key_value_heads,
        "dtype": model.config.dtype,
        "device": str(next(model.parameters()).device)
    }

def initialize_weights(model: nn.Module, config: VajraConfig):
    # Initialize weights according to GPT-style standards
    std = (2.0 / (config.hidden_size * 5)) ** 0.5
    for module in model.modules():
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()
