"""Hugging Face compatibility adapter layer for FoundationLM & VajraForCausalLM.

This module provides lightweight adapters that convert between Hugging Face
objects and native Vajra/FoundationLM implementations.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any

import torch

from model import VajraForCausalLM, VajraConfig, FoundationLM
from utils.file_utils import write_json
from utils.logging import setup_logger

logger = setup_logger("hf_compat")

try:
    from transformers import AutoConfig, AutoModelForCausalLM

    AutoConfig.register("vajra", VajraConfig)
    AutoConfig.register("vajra-lm", VajraConfig)
    AutoModelForCausalLM.register(VajraConfig, VajraForCausalLM)
except Exception:
    pass


def _model_config_to_hf_dict(cfg: VajraConfig) -> dict[str, Any]:
    """Convert a native VajraConfig to a Hugging Face config.json dict."""
    return {
        "architectures": ["VajraForCausalLM"],
        "model_type": "vajra",
        "hidden_size": cfg.hidden_size,
        "intermediate_size": cfg.intermediate_size,
        "max_position_embeddings": cfg.max_position_embeddings,
        "num_attention_heads": cfg.num_attention_heads,
        "num_hidden_layers": cfg.num_layers,
        "num_key_value_heads": cfg.num_key_value_heads,
        "rms_norm_eps": cfg.rms_norm_eps,
        "rope_theta": cfg.rope_theta,
        "tie_word_embeddings": cfg.tie_word_embeddings,
        "vocab_size": cfg.vocab_size,
        "torch_dtype": "float32",
        "transformers_version": "4.40.0",
    }


def _hf_dict_to_model_config(d: dict[str, Any]) -> VajraConfig:
    """Convert a Hugging Face config.json dict back to a native VajraConfig."""
    return VajraConfig(
        model_name=d.get("model_type", "vajra"),
        vocab_size=d["vocab_size"],
        hidden_size=d["hidden_size"],
        num_layers=d.get("num_hidden_layers", d.get("num_layers", 24)),
        num_attention_heads=d["num_attention_heads"],
        num_key_value_heads=d.get("num_key_value_heads", d["num_attention_heads"]),
        intermediate_size=d["intermediate_size"],
        max_position_embeddings=d["max_position_embeddings"],
        rope_theta=d.get("rope_theta", 10000.0),
        rms_norm_eps=d.get("rms_norm_eps", 1e-6),
        tie_word_embeddings=d.get("tie_word_embeddings", False),
    )


def save_pretrained(
    model: VajraForCausalLM | FoundationLM,
    output_dir: str | Path,
    tokenizer_dir: str | Path | None = None,
) -> dict[str, Any]:
    """Export a VajraForCausalLM/FoundationLM model to a Hugging Face-compatible directory.

    Generates:
      - config.json
      - generation_config.json
      - model.safetensors (via safetensors if available, else pytorch_model.bin)
      - tokenizer files (copied from tokenizer_dir if provided)
      - special_tokens_map.json
      - tokenizer_config.json

    Returns:
        Conversion report dict.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    cfg = model.config

    # 1. config.json
    hf_cfg = _model_config_to_hf_dict(cfg)
    write_json(hf_cfg, output_dir / "config.json")

    # 2. generation_config.json
    gen_config = {
        "max_new_tokens": 256,
        "do_sample": True,
        "temperature": 0.7,
        "top_p": 0.9,
        "top_k": 50,
        "repetition_penalty": 1.0,
        "eos_token_id": 3,
        "pad_token_id": 0,
    }
    write_json(gen_config, output_dir / "generation_config.json")

    # 3. Model weights — prefer safetensors
    state_dict = model.state_dict()

    # Handle tied weights: safetensors doesn't support shared tensors
    if cfg.tie_word_embeddings and "lm_head.weight" in state_dict:
        state_dict = {k: v for k, v in state_dict.items() if k != "lm_head.weight"}

    try:
        from safetensors.torch import save_file as st_save

        st_save(state_dict, str(output_dir / "model.safetensors"))
        weights_file = "model.safetensors"
    except ImportError:
        torch.save(state_dict, output_dir / "pytorch_model.bin")
        weights_file = "pytorch_model.bin"

    # 4. Tokenizer files
    if tokenizer_dir is not None:
        tok_src = Path(tokenizer_dir)
        for fname in ["tokenizer.json", "vocab.json", "merges.txt", "tokenizer.model"]:
            src = tok_src / fname
            if src.exists():
                shutil.copy2(src, output_dir / fname)

    # 5. special_tokens_map.json
    special_tokens = {
        "bos_token": "<s>",
        "eos_token": "</s>",
        "unk_token": "<unk>",
        "pad_token": "<pad>",
    }
    write_json(special_tokens, output_dir / "special_tokens_map.json")

    # 6. tokenizer_config.json
    tok_config = {
        "tokenizer_class": "PreTrainedTokenizerFast",
        "model_type": "vajra",
        "bos_token": "<s>",
        "eos_token": "</s>",
        "unk_token": "<unk>",
        "pad_token": "<pad>",
        "clean_up_tokenization_spaces": False,
    }
    write_json(tok_config, output_dir / "tokenizer_config.json")

    report = {
        "status": "success",
        "output_dir": str(output_dir),
        "weights_file": weights_file,
        "config_file": "config.json",
        "num_parameters": sum(p.numel() for p in model.parameters()),
        "files_created": sorted(str(f.name) for f in output_dir.iterdir() if f.is_file()),
    }

    write_json(report, output_dir / "conversion_report.json")
    logger.info(f"HF-compatible checkpoint saved to {output_dir}")
    return report


def load_pretrained(
    model_dir: str | Path,
    device: torch.device | str = "cpu",
    strict: bool = True,
) -> tuple[VajraForCausalLM, VajraConfig]:
    """Load a VajraForCausalLM model from a Hugging Face-compatible directory.

    Returns:
        Tuple of ``(model, config)``.
    """
    model_dir = Path(model_dir)

    # Load config.json -> VajraConfig
    config_path = model_dir / "config.json"
    if not config_path.exists():
        raise FileNotFoundError(f"config.json not found in {model_dir}")
    hf_dict = json.loads(config_path.read_text(encoding="utf-8"))
    model_cfg = _hf_dict_to_model_config(hf_dict)

    # Instantiate model
    model = VajraForCausalLM(model_cfg)

    # Load weights
    safetensors_path = model_dir / "model.safetensors"
    bin_path = model_dir / "pytorch_model.bin"

    if safetensors_path.exists():
        try:
            from safetensors.torch import load_file as st_load

            state_dict = st_load(str(safetensors_path), device=str(device))
        except ImportError:
            raise ImportError("safetensors package required to load model.safetensors")
    elif bin_path.exists():
        state_dict = torch.load(bin_path, map_location=device, weights_only=True)
    else:
        logger.warning(
            f"No weights file found in {model_dir}. Returning randomly initialized model."
        )
        return model.to(device), model_cfg

    if (
        model_cfg.tie_word_embeddings
        and "lm_head.weight" not in state_dict
        and "model.embed_tokens.weight" in state_dict
    ):
        state_dict["lm_head.weight"] = state_dict["model.embed_tokens.weight"]

    incompatible = model.load_state_dict(state_dict, strict=strict)
    if incompatible.missing_keys or incompatible.unexpected_keys:
        msg = f"State dict mismatch: missing keys={incompatible.missing_keys}, unexpected keys={incompatible.unexpected_keys}"
        if strict:
            raise RuntimeError(msg)
        else:
            logger.warning(msg)

    model = model.to(device)
    model.eval()

    logger.info(f"Loaded VajraForCausalLM from HF directory: {model_dir}")
    return model, model_cfg


def convert_checkpoint_to_hf(
    checkpoint_path: str | Path,
    model_config_path: str | Path,
    output_dir: str | Path,
    tokenizer_dir: str | Path | None = None,
) -> dict[str, Any]:
    """Convert a training checkpoint (.pt) to a Hugging Face directory."""
    from training.checkpoint import load_checkpoint

    model_cfg = VajraConfig.from_yaml(model_config_path)
    model = VajraForCausalLM(model_cfg)

    ckpt_path = Path(checkpoint_path)
    if ckpt_path.exists():
        load_checkpoint(ckpt_path, model)

    return save_pretrained(model, output_dir, tokenizer_dir)


def convert_hf_to_checkpoint(
    hf_dir: str | Path,
    output_path: str | Path,
) -> dict[str, Any]:
    """Convert a Hugging Face directory back to a training checkpoint (.pt)."""
    from training.checkpoint import save_checkpoint as _save_ckpt

    model, model_cfg = load_pretrained(hf_dir, device="cpu", strict=True)
    output_path = Path(output_path)
    _save_ckpt(output_path, model, step=0, tokens_seen=0)

    report = {
        "status": "success",
        "source": str(hf_dir),
        "output": str(output_path),
        "num_parameters": sum(p.numel() for p in model.parameters()),
    }
    logger.info(f"HF -> VajraForCausalLM checkpoint conversion complete: {output_path}")
    return report
