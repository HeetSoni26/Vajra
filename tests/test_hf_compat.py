"""Tests for Hugging Face compatibility adapter."""

from __future__ import annotations


import pytest
import torch

from model import FoundationLM, ModelConfig
from inference.hf_compat import (
    save_pretrained,
    load_pretrained,
    _model_config_to_hf_dict,
    _hf_dict_to_model_config,
    convert_checkpoint_to_hf,
    convert_hf_to_checkpoint,
)
from training.checkpoint import save_checkpoint, load_checkpoint


@pytest.fixture
def tiny_model():
    cfg = ModelConfig(
        model_name="hf-test",
        vocab_size=100,
        hidden_size=64,
        num_layers=2,
        num_attention_heads=2,
        num_key_value_heads=2,
        intermediate_size=128,
        max_position_embeddings=128,
    )
    return FoundationLM(cfg)


def test_config_roundtrip():
    """ModelConfig -> HF dict -> ModelConfig must preserve all values."""
    original = ModelConfig(
        vocab_size=200,
        hidden_size=128,
        num_layers=4,
        num_attention_heads=4,
        num_key_value_heads=2,
        intermediate_size=256,
        max_position_embeddings=512,
    )
    hf_dict = _model_config_to_hf_dict(original)
    restored = _hf_dict_to_model_config(hf_dict)

    assert restored.vocab_size == original.vocab_size
    assert restored.hidden_size == original.hidden_size
    assert restored.num_layers == original.num_layers
    assert restored.num_attention_heads == original.num_attention_heads
    assert restored.num_key_value_heads == original.num_key_value_heads
    assert restored.intermediate_size == original.intermediate_size
    assert restored.max_position_embeddings == original.max_position_embeddings


def test_save_and_load_pretrained(tiny_model, tmp_path):
    """save_pretrained -> load_pretrained round-trip preserves model weights."""
    hf_dir = tmp_path / "hf_export"

    # Save
    report = save_pretrained(tiny_model, hf_dir)
    assert report["status"] == "success"
    assert (hf_dir / "config.json").exists()
    assert (hf_dir / "generation_config.json").exists()

    # Load
    loaded_model, loaded_cfg = load_pretrained(hf_dir)

    # Verify weight equality
    for (n1, p1), (n2, p2) in zip(
        tiny_model.named_parameters(), loaded_model.named_parameters()
    ):
        assert n1 == n2
        assert torch.allclose(p1, p2, atol=1e-6), f"Weight mismatch at {n1}"


def test_checkpoint_roundtrip_conversion(tiny_model, tmp_path):
    """FoundationLM ckpt -> HF dir -> FoundationLM ckpt round-trip."""
    # Save a training checkpoint
    ckpt_path = tmp_path / "original.pt"
    save_checkpoint(ckpt_path, tiny_model, step=10, tokens_seen=1000)

    # Write a temporary model config YAML
    model_yaml = tmp_path / "model_cfg.yaml"
    import yaml
    cfg_dict = {
        "model_name": "hf-test",
        "vocab_size": 100,
        "hidden_size": 64,
        "num_layers": 2,
        "num_attention_heads": 2,
        "num_key_value_heads": 2,
        "intermediate_size": 128,
        "max_position_embeddings": 128,
    }
    model_yaml.write_text(yaml.dump(cfg_dict))

    # Convert ckpt -> HF
    hf_dir = tmp_path / "hf_roundtrip"
    convert_checkpoint_to_hf(ckpt_path, model_yaml, hf_dir)
    assert (hf_dir / "config.json").exists()

    # Convert HF -> ckpt
    restored_ckpt = tmp_path / "restored.pt"
    convert_hf_to_checkpoint(hf_dir, restored_ckpt)
    assert restored_ckpt.exists()

    # Load both and compare weights
    original = FoundationLM(ModelConfig(**cfg_dict))
    load_checkpoint(ckpt_path, original)

    restored = FoundationLM(ModelConfig(**cfg_dict))
    load_checkpoint(restored_ckpt, restored)

    for (n1, p1), (n2, p2) in zip(
        original.named_parameters(), restored.named_parameters()
    ):
        assert torch.allclose(p1, p2, atol=1e-6), f"Roundtrip mismatch at {n1}"
