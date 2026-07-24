"""Tests for the evaluation framework."""

from __future__ import annotations


import numpy as np
import pytest
import torch

from evaluation.evaluator import ModelEvaluator
from model import FoundationLM, ModelConfig


@pytest.fixture
def tiny_model():
    cfg = ModelConfig(
        model_name="eval-test",
        vocab_size=100,
        hidden_size=64,
        num_layers=2,
        num_attention_heads=2,
        num_key_value_heads=2,
        intermediate_size=128,
        max_position_embeddings=128,
    )
    return FoundationLM(cfg)


def test_evaluate_dataset(tiny_model, tmp_path):
    """Evaluate on a synthetic memmap dataset."""
    # Create synthetic data
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    tokens = np.random.randint(0, 100, size=500, dtype=np.uint32)
    tokens.tofile(data_dir / "val.bin")

    evaluator = ModelEvaluator(tiny_model, torch.device("cpu"))
    result = evaluator.evaluate_dataset(
        data_dir / "val.bin", sequence_length=32, batch_size=2, max_batches=5
    )

    assert "avg_cross_entropy" in result
    assert "perplexity" in result
    assert "bits_per_character" in result
    assert "tokens_per_sec" in result
    assert result["perplexity"] > 0


def test_evaluate_all(tiny_model, tmp_path):
    """Evaluate on all splits."""
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    for name in ["train.bin", "val.bin", "test.bin"]:
        tokens = np.random.randint(0, 100, size=500, dtype=np.uint32)
        tokens.tofile(data_dir / name)

    evaluator = ModelEvaluator(tiny_model, torch.device("cpu"))
    report = evaluator.evaluate_all(data_dir, sequence_length=32, batch_size=2, max_batches=3)

    assert "splits" in report
    assert len(report["splits"]) == 3
    assert "model_info" in report
