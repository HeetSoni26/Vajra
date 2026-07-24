import tempfile
from pathlib import Path

import torch

from model.checkpoints import CheckpointManager
from model.config import VajraConfig
from model.generation.engine import GenerationEngine
from model.layers.attention import VajraAttention
from model.layers.mlp import VajraMLP
from model.layers.rmsnorm import RMSNorm
from model.layers.rope import RotaryEmbedding
from model.modeling import VajraForCausalLM
from model.utils import summarize_model


def test_rmsnorm():
    norm = RMSNorm(hidden_size=16)
    x = torch.randn(2, 4, 16)
    out = norm(x)
    assert out.shape == x.shape
    assert not torch.isnan(out).any()


def test_rope():
    emb = RotaryEmbedding(dim=16, max_position_embeddings=128)
    x = torch.randn(2, 10, 8, 16)
    cos, sin = emb(x, seq_len=10)
    assert cos.shape == (10, 16)
    assert sin.shape == (10, 16)


def test_attention():
    config = VajraConfig(hidden_size=64, num_attention_heads=4, num_key_value_heads=2)
    attn = VajraAttention(config)
    x = torch.randn(2, 10, 64)
    out, _, _ = attn(hidden_states=x, attention_mask=None, position_ids=None)
    assert out.shape == (2, 10, 64)


def test_mlp():
    config = VajraConfig(hidden_size=64, intermediate_size=128)
    mlp = VajraMLP(config)
    x = torch.randn(2, 10, 64)
    out = mlp(x)
    assert out.shape == (2, 10, 64)


def test_model_forward():
    # Miniature config
    config = VajraConfig(
        vocab_size=100,
        hidden_size=64,
        intermediate_size=128,
        num_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
    )
    model = VajraForCausalLM(config)
    input_ids = torch.randint(0, 100, (2, 10))

    outputs = model(input_ids)
    assert "logits" in outputs
    assert outputs["logits"].shape == (2, 10, 100)


def test_generation():
    config = VajraConfig(
        vocab_size=100,
        hidden_size=64,
        intermediate_size=128,
        num_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
    )
    model = VajraForCausalLM(config)
    engine = GenerationEngine(model)

    input_ids = torch.randint(0, 100, (1, 5))
    out = engine.generate(input_ids, max_new_tokens=5)
    assert out.shape == (1, 10)


def test_checkpoints():
    config = VajraConfig(
        vocab_size=100,
        hidden_size=64,
        intermediate_size=128,
        num_layers=1,
        num_attention_heads=2,
        num_key_value_heads=1,
    )
    model = VajraForCausalLM(config)

    with tempfile.TemporaryDirectory() as d:
        CheckpointManager.save_checkpoint(model, d, use_safetensors=False)
        assert (Path(d) / "pytorch_model.bin").exists()
        assert (Path(d) / "config.json").exists()

        loaded = CheckpointManager.load_checkpoint(d)
        assert summarize_model(loaded)["parameters"] == summarize_model(model)["parameters"]


def test_label_shifting():
    config = VajraConfig(
        vocab_size=100,
        hidden_size=64,
        intermediate_size=128,
        num_layers=1,
        num_attention_heads=2,
        num_key_value_heads=1,
    )
    model = VajraForCausalLM(config)
    input_ids = torch.randint(0, 100, (2, 10))
    labels = torch.randint(0, 100, (2, 10))
    out = model(input_ids, labels=labels)
    assert "loss" in out and out["loss"] is not None
