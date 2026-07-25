"""Verification script for validating model loading and generation from the release package."""

import json
from pathlib import Path
import torch
from tokenizers import Tokenizer
from inference.engine import InferenceEngine, GenerationConfig
from model import FoundationLM, ModelConfig


def verify_model_package(package_dir: str | Path = "release/vajra-57m") -> None:
    package_path = Path(package_dir)
    print(f"Verifying local package loading from: {package_path.resolve()}")

    # 1. Config loading
    config_file = package_path / "config.json"
    assert config_file.exists(), "config.json is missing"
    with config_file.open("r", encoding="utf-8") as f:
        cfg_dict = json.load(f)
    print("[PASS] Successfully loaded config.json")

    # Map keys to ModelConfig
    model_cfg = ModelConfig(
        vocab_size=cfg_dict.get("vocab_size", 65536),
        hidden_size=cfg_dict.get("hidden_size", 512),
        intermediate_size=cfg_dict.get("intermediate_size", 1376),
        num_layers=cfg_dict.get("num_layers", 8),
        num_attention_heads=cfg_dict.get("num_attention_heads", 8),
        num_key_value_heads=cfg_dict.get("num_key_value_heads", 4),
        max_position_embeddings=cfg_dict.get("max_position_embeddings", 2048),
        tie_word_embeddings=cfg_dict.get("tie_word_embeddings", True),
    )

    # 2. Architecture instantiation
    model = FoundationLM(model_cfg)
    print("[PASS] Successfully instantiated FoundationLM architecture")

    # 3. Weights loading
    weights_safetensors = package_path / "model.safetensors"
    weights_bin = package_path / "pytorch_model.bin"

    if weights_safetensors.exists():
        from safetensors.torch import load_file
        state_dict = load_file(weights_safetensors)
        model.load_state_dict(state_dict)
        print("[PASS] Successfully loaded weights from model.safetensors")
    elif weights_bin.exists():
        state_dict = torch.load(weights_bin, map_location="cpu")
        model.load_state_dict(state_dict)
        print("[PASS] Successfully loaded weights from pytorch_model.bin")
    else:
        raise FileNotFoundError("Neither model.safetensors nor pytorch_model.bin found")

    model.eval()

    # 4. Generation test
    tokenizer = Tokenizer.from_file("tokenizer/v1.0/tokenizer.json")
    engine = InferenceEngine(model=model, tokenizer=tokenizer, device=torch.device("cpu"))
    sample_prompt = "Vajra foundation model pretraining"
    gen_config = GenerationConfig(max_new_tokens=20, temperature=0.7)
    output_text = engine.generate(sample_prompt, gen_cfg=gen_config)
    print(f"[PASS] Generation Output for '{sample_prompt}':\n  -> {output_text}")
    print("\nModel package loading and generation verification completed successfully!")


if __name__ == "__main__":
    verify_model_package()
