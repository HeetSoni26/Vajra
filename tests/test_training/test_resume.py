import random
from pathlib import Path

import pytest
import torch
from torch import nn

from training.resume import CheckpointValidationError, ResumeManager


class DummyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(10, 10)


@pytest.fixture
def temp_exp_dir(tmp_path):
    base_dir = tmp_path / "checkpoints"
    base_dir.mkdir()
    return base_dir


def create_dummy_checkpoint(
    exp_dir: Path, step: int, corrupt: bool = False, missing_keys: bool = False
):
    ckpt_path = exp_dir / "latest.pt"

    if corrupt:
        ckpt_path.write_text("corrupted data")
        return ckpt_path

    model = DummyModel()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)

    state = {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": None,
        "step": step,
        "tokens_seen": step * 100,
        "rng_state": {"python": random.getstate()},
    }

    if missing_keys:
        del state["optimizer"]

    torch.save(state, ckpt_path)
    return ckpt_path


def test_discover_experiments(temp_exp_dir):
    manager = ResumeManager(temp_exp_dir)

    (temp_exp_dir / "exp_2").mkdir()
    (temp_exp_dir / "exp_1").mkdir()
    (temp_exp_dir / "exp_3").mkdir()

    exps = manager.discover_experiments()
    assert len(exps) == 3
    assert exps[0].name == "exp_3"
    assert exps[1].name == "exp_2"
    assert exps[2].name == "exp_1"


def test_resume_latest_checkpoint(temp_exp_dir):
    manager = ResumeManager(temp_exp_dir)

    exp1 = temp_exp_dir / "run_1"
    exp1.mkdir()
    create_dummy_checkpoint(exp1, step=10)

    exp_dir, state = manager.find_latest_valid_experiment(prefix="run_")

    assert exp_dir.name == "run_1"
    assert state["step"] == 10
    assert state["tokens_seen"] == 1000


def test_resume_after_deleted_latest(temp_exp_dir):
    manager = ResumeManager(temp_exp_dir)

    exp1 = temp_exp_dir / "run_1"
    exp1.mkdir()
    create_dummy_checkpoint(exp1, step=10)

    exp2 = temp_exp_dir / "run_2"
    exp2.mkdir()
    # exp2 has no latest.pt

    exp_dir, state = manager.find_latest_valid_experiment(prefix="run_")

    # Should fallback to run_1
    assert exp_dir.name == "run_1"
    assert state["step"] == 10


def test_resume_after_corrupted_checkpoint(temp_exp_dir):
    manager = ResumeManager(temp_exp_dir)

    exp1 = temp_exp_dir / "run_1"
    exp1.mkdir()
    create_dummy_checkpoint(exp1, step=10)

    exp2 = temp_exp_dir / "run_2"
    exp2.mkdir()
    create_dummy_checkpoint(exp2, step=20, corrupt=True)

    exp_dir, state = manager.find_latest_valid_experiment(prefix="run_")

    # run_2 is corrupted, should fallback to run_1
    assert exp_dir.name == "run_1"
    assert state["step"] == 10


def test_checkpoint_validation_missing_keys(temp_exp_dir):
    manager = ResumeManager(temp_exp_dir)

    exp1 = temp_exp_dir / "run_1"
    exp1.mkdir()
    ckpt_path = create_dummy_checkpoint(exp1, step=10, missing_keys=True)

    with pytest.raises(CheckpointValidationError):
        manager.validate_checkpoint(ckpt_path)


def test_restore_state(temp_exp_dir):
    manager = ResumeManager(temp_exp_dir)

    exp1 = temp_exp_dir / "run_1"
    exp1.mkdir()
    create_dummy_checkpoint(exp1, step=15)

    exp_dir, state = manager.find_latest_valid_experiment(prefix="run_")

    model = DummyModel()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)

    start_step, tokens = manager.restore_state(state, model, optimizer)

    assert start_step == 15
    assert tokens == 1500
