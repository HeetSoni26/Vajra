import torch
from pathlib import Path
from typing import Union

# Attempt to load safetensors if available
try:
    from safetensors.torch import save_file, load_file
    HAS_SAFETENSORS = True
except ImportError:
    HAS_SAFETENSORS = False

from model.modeling import VajraForCausalLM
from model.config import VajraConfig

class CheckpointManager:
    """
    Manages safe serialization and loading of Vajra model checkpoints.
    Supports safetensors natively.
    """
    
    @staticmethod
    def save_checkpoint(model: VajraForCausalLM, output_dir: Union[str, Path], use_safetensors: bool = True):
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Save config
        model.config.save_pretrained(output_dir)
        
        # Extract state dict
        state_dict = model.state_dict()
        
        # Optional safetensors support
        if use_safetensors and HAS_SAFETENSORS:
            save_file(state_dict, output_dir / "model.safetensors", metadata={"format": "pt"})
        else:
            torch.save(state_dict, output_dir / "pytorch_model.bin")
            
    @staticmethod
    def load_checkpoint(output_dir: Union[str, Path], device: str = "cpu") -> VajraForCausalLM:
        output_dir = Path(output_dir)
        if not output_dir.exists():
            raise FileNotFoundError(f"Checkpoint directory {output_dir} not found.")
            
        # Load config
        config = VajraConfig.from_pretrained(output_dir)
        config.device = device
        
        # Initialize bare model
        model = VajraForCausalLM(config)
        model.to(device)
        
        # Load state
        safetensors_path = output_dir / "model.safetensors"
        bin_path = output_dir / "pytorch_model.bin"
        
        if safetensors_path.exists() and HAS_SAFETENSORS:
            state_dict = load_file(safetensors_path, device=device)
        elif bin_path.exists():
            state_dict = torch.load(bin_path, map_location=device, weights_only=True)
        else:
            raise FileNotFoundError(f"No checkpoint found in {output_dir}")
            
        model.load_state_dict(state_dict)
        return model
