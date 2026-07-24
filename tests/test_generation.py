"""Comprehensive tests for the inference engine, generation, KV cache, and sampling."""

from __future__ import annotations

import pytest
import torch

from inference.kv_cache import KVCache
from inference.engine import (
    InferenceEngine,
    GenerationConfig,
    _apply_repetition_penalty,
    _apply_top_k,
    _apply_top_p,
    _sample_next_token,
)
from model import FoundationLM, ModelConfig


@pytest.fixture
def tiny_model():
    cfg = ModelConfig(
        model_name="test-gen",
        vocab_size=100,
        hidden_size=64,
        num_layers=2,
        num_attention_heads=2,
        num_key_value_heads=2,
        intermediate_size=128,
        max_position_embeddings=128,
    )
    return FoundationLM(cfg)


# --- KV Cache Tests ---


def test_kv_cache_update_and_reset():
    cache = KVCache(num_layers=2, max_batch_size=1, max_seq_len=64)
    assert cache.seq_len == 0

    k = torch.randn(1, 2, 4, 32)
    v = torch.randn(1, 2, 4, 32)
    k_out, v_out = cache.update(0, k, v)

    assert k_out.shape == (1, 2, 4, 32)
    assert cache.seq_len == 4

    # Append more
    k2 = torch.randn(1, 2, 1, 32)
    v2 = torch.randn(1, 2, 1, 32)
    k_out2, v_out2 = cache.update(0, k2, v2)
    assert k_out2.shape == (1, 2, 5, 32)
    assert cache.seq_len == 5

    cache.reset()
    assert cache.seq_len == 0
    assert cache.get(0) == (None, None)


def test_kv_cache_batch():
    cache = KVCache(num_layers=2, max_batch_size=2, max_seq_len=64)
    k = torch.randn(2, 2, 8, 32)
    v = torch.randn(2, 2, 8, 32)
    k_out, v_out = cache.update(0, k, v)
    assert k_out.shape[0] == 2


# --- Sampling Tests ---


def test_greedy_decoding():
    logits = torch.tensor([[1.0, 5.0, 2.0]])
    gen_cfg = GenerationConfig(do_sample=False)
    token = _sample_next_token(logits, gen_cfg)
    assert token.item() == 1  # index of max value


def test_temperature_sampling():
    torch.manual_seed(42)
    logits = torch.randn(1, 100)
    gen_cfg = GenerationConfig(temperature=0.5, do_sample=True)
    token = _sample_next_token(logits.clone(), gen_cfg)
    assert 0 <= token.item() < 100


def test_top_k_filtering():
    logits = torch.tensor([[10.0, 9.0, 8.0, 7.0, 6.0]])
    filtered = _apply_top_k(logits.clone(), k=3)
    assert (filtered[0, 3:] == float("-inf")).all()
    assert filtered[0, 0] == 10.0


def test_top_p_filtering():
    logits = torch.tensor([[10.0, 5.0, 1.0, 0.1, 0.01]])
    filtered = _apply_top_p(logits.clone(), p=0.9)
    # Top-p should keep the highest-probability tokens
    assert filtered[0, 0] == 10.0


def test_repetition_penalty():
    logits = torch.tensor([[5.0, 3.0, 1.0]])
    generated = torch.tensor([[0, 1]])
    penalized = _apply_repetition_penalty(logits.clone(), generated, penalty=2.0)
    assert penalized[0, 0] < 5.0  # Token 0 was generated, should be penalized
    assert penalized[0, 2] == 1.0  # Token 2 was not generated, unchanged


# --- Forward Pass with KV Cache ---


def test_model_forward_with_kv_cache(tiny_model):
    model = tiny_model
    model.eval()
    input_ids = torch.randint(0, 100, (1, 8))

    # Without cache
    out1 = model(input_ids)
    assert "logits" in out1
    assert out1["logits"].shape == (1, 8, 100)

    # With cache — prefill
    cache = KVCache(num_layers=2, max_batch_size=1, max_seq_len=64)
    out2 = model(input_ids, kv_cache=cache, start_pos=0)
    assert out2["logits"].shape == (1, 8, 100)
    assert cache.seq_len == 8

    # Decode — single token
    next_token = torch.randint(0, 100, (1, 1))
    out3 = model(next_token, kv_cache=cache, start_pos=8)
    assert out3["logits"].shape == (1, 1, 100)
    assert cache.seq_len == 9

    cache.reset()


def test_kv_cache_correctness(tiny_model):
    """Verify that KV-cached generation matches full-context recomputation."""
    model = tiny_model
    model.eval()
    torch.manual_seed(42)

    input_ids = torch.randint(0, 100, (1, 4))

    # Full forward (no cache) with 4+1 tokens
    extra = torch.randint(0, 100, (1, 1))
    full_input = torch.cat([input_ids, extra], dim=1)
    with torch.no_grad():
        out_full = model(full_input)
    logits_full = out_full["logits"][:, -1, :]

    # Cached forward: prefill 4, then decode 1
    cache = KVCache(num_layers=2, max_batch_size=1, max_seq_len=64)
    with torch.no_grad():
        model(input_ids, kv_cache=cache, start_pos=0)
        out_cached = model(extra, kv_cache=cache, start_pos=4)
    logits_cached = out_cached["logits"][:, -1, :]

    # Logits must match
    assert torch.allclose(logits_full, logits_cached, atol=1e-4), (
        f"Max diff: {(logits_full - logits_cached).abs().max().item()}"
    )

    cache.reset()


# --- Batch Generation ---


def test_batch_generation(tiny_model):
    """Test generating from multiple prompts simultaneously."""
    from unittest.mock import MagicMock

    tok = MagicMock()
    tok.encode = lambda text: MagicMock(ids=list(range(len(text) % 10 + 2)))
    tok.decode = lambda ids: "generated"
    tok.eos_token_id = None

    engine = InferenceEngine(tiny_model, tok, torch.device("cpu"))
    results = engine.generate(
        ["Hello", "World"], GenerationConfig(max_new_tokens=5, do_sample=False, use_kv_cache=True)
    )
    assert len(results) == 2


# --- Streaming Generation ---


def test_streaming_generation(tiny_model):
    """Test streaming token-by-token generation."""
    from unittest.mock import MagicMock

    tok = MagicMock()
    tok.encode = lambda text: MagicMock(ids=[1, 2, 3])
    tok.decode = lambda ids: "t"
    tok.eos_token_id = None

    engine = InferenceEngine(tiny_model, tok, torch.device("cpu"))
    gen_cfg = GenerationConfig(max_new_tokens=5, do_sample=False, use_kv_cache=True)

    tokens = list(engine.generate_stream("Hello", gen_cfg))
    assert len(tokens) == 5


# --- Backward Compatibility ---


def test_training_forward_still_works(tiny_model):
    """Ensure the model forward pass for training (no KV cache) still works."""
    model = tiny_model
    model.train()
    input_ids = torch.randint(0, 100, (2, 16))
    labels = torch.randint(0, 100, (2, 16))

    out = model(input_ids, labels=labels)
    assert "loss" in out
    assert "logits" in out
    out["loss"].backward()
