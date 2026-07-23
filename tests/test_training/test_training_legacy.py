from pathlib import Path

import numpy as np
import pytest
import torch

from model import FoundationLM, ModelConfig
from training.data_loader import create_dataloaders
from training.optimizer import build_optimizer, cosine_lr
from training.trainer import Trainer


@pytest.fixture
def tiny_model_cfg() -> ModelConfig:
    return ModelConfig(
        model_name="test-tiny",
        vocab_size=100,
        hidden_size=64,
        num_layers=2,
        num_attention_heads=2,
        num_key_value_heads=2,
        intermediate_size=128,
        max_position_embeddings=128,
    )


@pytest.fixture
def mock_token_data(tmp_path: Path) -> Path:
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    tokens = np.arange(1000, dtype=np.uint32) % 100
    tokens.tofile(data_dir / "train.bin")
    tokens[:200].tofile(data_dir / "val.bin")
    return data_dir


def test_memmap_dataset_and_dataloaders(mock_token_data: Path):
    train_loader, val_loader = create_dataloaders(mock_token_data, sequence_length=32, micro_batch_size=4)
    assert train_loader is not None
    assert val_loader is not None

    batch = next(iter(train_loader))
    assert batch["input_ids"].shape == (4, 32)
    assert batch["labels"].shape == (4, 32)
    # Target label is right-shifted input_id
    assert torch.equal(batch["input_ids"][0, 1:], batch["labels"][0, :-1])


def test_optimizer_and_cosine_scheduler(tiny_model_cfg: ModelConfig):
    model = FoundationLM(tiny_model_cfg)
    optimizer = build_optimizer(model, lr=1e-3, weight_decay=0.01)

    assert len(optimizer.param_groups) == 2
    assert optimizer.param_groups[0]["weight_decay"] == 0.01
    assert optimizer.param_groups[1]["weight_decay"] == 0.0

    lr_warmup = cosine_lr(step=5, warmup_steps=10, total_steps=100, peak_lr=1e-3, min_lr=1e-4)
    assert lr_warmup < 1e-3

    lr_decay = cosine_lr(step=50, warmup_steps=10, total_steps=100, peak_lr=1e-3, min_lr=1e-4)
    assert lr_decay < 1e-3


def test_trainer_step_and_early_failure(tiny_model_cfg: ModelConfig):
    model = FoundationLM(tiny_model_cfg)
    optimizer = build_optimizer(model, lr=1e-3, weight_decay=0.01)
    trainer = Trainer(model, optimizer, grad_clip=1.0, grad_accum_steps=1, warmup_steps=5, total_steps=20)

    batch = {
        "input_ids": torch.randint(0, 100, (2, 32)),
        "labels": torch.randint(0, 100, (2, 32)),
    }

    step1 = trainer.train_step(batch, step=0, is_accum_step=True)
    assert step1 is not None
    assert "loss" in step1
    assert step1["loss"] > 0

    # Early Failure NaN Check
    with pytest.raises(ValueError, match="Loss is NaN/Inf"):
        bad_batch = {
            "input_ids": torch.randint(0, 100, (2, 32)),
            "labels": torch.randint(0, 100, (2, 32)),
        }

        def nan_forward(*args, **kwargs):
            return {"loss": torch.tensor(float("nan"))}

        model.forward = nan_forward
        trainer.train_step(bad_batch, step=1, is_accum_step=True)


def test_trainer_evaluation(tiny_model_cfg: ModelConfig, mock_token_data: Path):
    model = FoundationLM(tiny_model_cfg)
    optimizer = build_optimizer(model, lr=1e-3, weight_decay=0.01)
    trainer = Trainer(model, optimizer)

    _, val_loader = create_dataloaders(mock_token_data, sequence_length=32, micro_batch_size=4)
    assert val_loader is not None
    val_stats = trainer.evaluate(val_loader)

    assert "val_loss" in val_stats
    assert "val_perplexity" in val_stats
    assert val_stats["val_loss"] >= 0
