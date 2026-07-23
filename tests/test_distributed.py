from __future__ import annotations

import os
from pathlib import Path

import pytest
import torch

from model import FoundationLM, ModelConfig
from training.data_loader import create_dataloaders
from training.pretrain import cleanup_ddp_environment, setup_ddp_environment
from training.trainer import resolve_precision_and_scaler


@pytest.fixture
def model_cfg() -> ModelConfig:
    return ModelConfig(
        model_name="dist-test",
        vocab_size=100,
        hidden_size=64,
        num_layers=2,
        num_attention_heads=2,
        num_key_value_heads=2,
        intermediate_size=128,
        max_position_embeddings=128,
    )


def test_distributed_sampler_integration(tmp_path: Path):
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    tokens = (torch.arange(500) % 100).numpy().astype("uint32")
    tokens.tofile(data_dir / "train.bin")

    train_loader, _ = create_dataloaders(
        data_dir=data_dir,
        sequence_length=32,
        micro_batch_size=2,
        is_distributed=True,
        world_size=2,
        rank=0,
    )
    assert train_loader is not None
    assert hasattr(train_loader, "sampler")


def test_precision_and_scaler_resolution():
    dtype, enabled, scaler = resolve_precision_and_scaler("fp32")
    assert dtype == torch.float32
    assert not enabled

    dtype_bf, _, _ = resolve_precision_and_scaler("bf16")
    if torch.cuda.is_available() and torch.cuda.is_bf16_supported():
        assert dtype_bf == torch.bfloat16
    else:
        assert dtype_bf == torch.float32


def test_gradient_checkpointing_numerical_equivalence(model_cfg: ModelConfig):
    torch.manual_seed(1337)
    cfg_standard = ModelConfig(**{**model_cfg.model_dump(), "use_gradient_checkpointing": False})
    model_std = FoundationLM(cfg_standard)

    torch.manual_seed(1337)
    cfg_ckpt = ModelConfig(**{**model_cfg.model_dump(), "use_gradient_checkpointing": True})
    model_ckpt = FoundationLM(cfg_ckpt)

    input_ids = torch.randint(0, 100, (2, 16))
    labels = torch.randint(0, 100, (2, 16))

    out_std = model_std(input_ids, labels=labels)
    out_ckpt = model_ckpt(input_ids, labels=labels)

    # Forward loss equivalence
    assert torch.allclose(out_std["loss"], out_ckpt["loss"], atol=1e-5)

    # Backward gradient equivalence
    out_std["loss"].backward()
    out_ckpt["loss"].backward()

    for p_std, p_ckpt in zip(model_std.parameters(), model_ckpt.parameters()):
        if p_std.grad is not None and p_ckpt.grad is not None:
            assert torch.allclose(p_std.grad, p_ckpt.grad, atol=1e-4)


def test_ddp_environment_setup_and_teardown():
    os.environ["WORLD_SIZE"] = "1"
    os.environ["RANK"] = "0"
    os.environ["LOCAL_RANK"] = "0"

    is_dist, w_size, rank, l_rank = setup_ddp_environment()
    assert not is_dist  # WORLD_SIZE=1 should not trigger multi-process DDP
    assert w_size == 1

    cleanup_ddp_environment(is_dist)
