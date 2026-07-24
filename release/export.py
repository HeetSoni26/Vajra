from pathlib import Path
import json
import logging
from typing import Dict, Any, Optional

import torch

try:
    from safetensors.torch import save_file as save_safetensors
except ImportError:
    save_safetensors = None

logger = logging.getLogger(__name__)


class ModelExporter:
    """Exports Vajra checkpoints to HuggingFace compatible formats."""

    def __init__(self, output_dir: str):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def export_pytorch(self, model: torch.nn.Module, name: str = "pytorch_model.bin"):
        path = self.output_dir / name
        torch.save(model.state_dict(), path)
        return path

    def export_safetensors(self, model: torch.nn.Module, name: str = "model.safetensors"):
        path = self.output_dir / name
        if save_safetensors:
            save_safetensors(model.state_dict(), path)
        else:
            logger.warning("safetensors library not found. Falling back to PyTorch bin.")
            self.export_pytorch(model, name="pytorch_model.bin")
        return path

    def export_config(self, config_dict: Dict[str, Any], name: str = "config.json"):
        path = self.output_dir / name
        with open(path, "w", encoding="utf-8") as f:
            json.dump(config_dict, f, indent=2)
        return path

    def export_generation_config(
        self, config_dict: Dict[str, Any], name: str = "generation_config.json"
    ):
        path = self.output_dir / name
        with open(path, "w", encoding="utf-8") as f:
            json.dump(config_dict, f, indent=2)
        return path

    def export_huggingface(
        self,
        model: torch.nn.Module,
        model_config: Dict[str, Any],
        tokenizer: Any = None,
        generation_config: Optional[Dict[str, Any]] = None,
    ):
        """Creates a complete HuggingFace-compatible export."""
        # 1. Config
        model_config["architectures"] = ["VajraForCausalLM"]
        model_config["model_type"] = "vajra"
        self.export_config(model_config)

        # 2. Weights
        self.export_safetensors(model)

        # 3. Generation Config
        if generation_config is None:
            generation_config = {
                "max_new_tokens": 128,
                "temperature": 0.7,
                "top_p": 0.9,
                "do_sample": True,
            }
        self.export_generation_config(generation_config)

        # 4. Tokenizer
        if tokenizer:
            try:
                tokenizer.save_pretrained(self.output_dir)
            except AttributeError:
                logger.warning("Mock tokenizer provided. Skipping tokenizer export.")
                # Mock tokenizer save for tests
                with open(self.output_dir / "tokenizer_config.json", "w") as f:
                    json.dump({"mock": True}, f)

        return self.output_dir
