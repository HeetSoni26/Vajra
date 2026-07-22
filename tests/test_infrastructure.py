from __future__ import annotations

from pathlib import Path

import torch
import torch.nn as nn

from training.checkpoint import CheckpointManager
from utils.config import apply_cli_overrides, merge_configs
from utils.environment import get_device, get_memory_info, set_seed


class SimpleModel(nn.Module):

    def __init__(self) -> None:
        super().__init__()
        self.fc = nn.Linear(4, 2)


def test_seed_reproducibility():
    set_seed(1337)
    val1 = torch.randn(2, 2)
    set_seed(1337)
    val2 = torch.randn(2, 2)
    assert torch.equal(val1, val2)


def test_device_and_memory():
    device = get_device()
    assert isinstance(device, torch.device)
    mem_info = get_memory_info()
    assert isinstance(mem_info, dict)


def test_config_merging_and_cli_overrides():
    base = {"model": {"hidden_size": 256, "num_layers": 4}, "lr": 0.001}
    override = {"model": {"num_layers": 8}, "lr": 0.0005}
    merged = merge_configs(base, override)
    assert merged["model"]["num_layers"] == 8
    assert merged["model"]["hidden_size"] == 256

    cli_applied = apply_cli_overrides(merged, ["model.hidden_size=512", "lr=0.0001", "debug=true"])
    assert cli_applied["model"]["hidden_size"] == 512
    assert cli_applied["lr"] == 0.0001
    assert cli_applied["debug"] is True


def test_checkpoint_manager(tmp_path: Path):
    model = SimpleModel()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    manager = CheckpointManager(checkpoint_dir=tmp_path / "checkpoints", max_to_keep=2, metric_name="loss", mode="min")

    manager.save(step=10, model=model, optimizer=optimizer, tokens_seen=1000, metrics={"loss": 2.5})
    manager.save(step=20, model=model, optimizer=optimizer, tokens_seen=2000, metrics={"loss": 1.5})
    manager.save(step=30, model=model, optimizer=optimizer, tokens_seen=3000, metrics={"loss": 1.8})

    assert (tmp_path / "checkpoints" / "latest.pt").exists()
    assert (tmp_path / "checkpoints" / "best.pt").exists()

    # Verify best checkpoint recorded step 20 (lowest loss = 1.5)
    best_state = manager.load_best(model, optimizer)
    assert best_state["step"] == 20
    assert best_state["metrics"]["loss"] == 1.5

    # Verify max_to_keep pruning
    step10_exists = (tmp_path / "checkpoints" / "checkpoint_step_10.pt").exists()
    assert not step10_exists  # Should have been pruned as max_to_keep=2
