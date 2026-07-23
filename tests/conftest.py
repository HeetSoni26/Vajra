"""Global Pytest fixtures and shared configuration for Vajra test suite."""

from __future__ import annotations

from pathlib import Path
import pytest
import torch

from model.config import VajraConfig
from model.modeling import VajraForCausalLM


@pytest.fixture
def tiny_config() -> VajraConfig:
    """Fixture providing a tiny, fast VajraConfig for unit testing."""
    return VajraConfig(
        vocab_size=100,
        hidden_size=64,
        intermediate_size=128,
        num_layers=2,
        num_attention_heads=2,
        num_key_value_heads=2,
        max_position_embeddings=128,
        model_name="test-tiny",
    )


@pytest.fixture
def tiny_model(tiny_config: VajraConfig) -> VajraForCausalLM:
    """Fixture providing an initialized tiny VajraForCausalLM model."""
    torch.manual_seed(42)
    return VajraForCausalLM(tiny_config)


@pytest.fixture
def temp_work_dir(tmp_path: Path) -> Path:
    """Fixture providing a clean temporary directory for filesystem testing."""
    work_dir = tmp_path / "work_dir"
    work_dir.mkdir(parents=True, exist_ok=True)
    return work_dir
