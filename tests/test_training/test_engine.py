import tempfile
from pathlib import Path

import torch

from model.config import VajraConfig
from model.modeling import VajraForCausalLM
from training.config import TrainingConfig
from training.optim.optimizers import create_optimizer
from training.optim.schedulers import create_scheduler


def test_optimizer():
    config = VajraConfig(
        vocab_size=100,
        hidden_size=64,
        intermediate_size=128,
        num_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
    )
    model = VajraForCausalLM(config)
    t_config = TrainingConfig()

    opt = create_optimizer(model, t_config)
    assert len(opt.param_groups) == 2
    # Check weight decay is 0 for biases and layernorms
    no_decay = opt.param_groups[1]
    assert no_decay["weight_decay"] == 0.0


def test_scheduler_linear():
    model = torch.nn.Linear(10, 10)
    t_config = TrainingConfig(lr_scheduler_type="linear", warmup_steps=10, max_steps=100)
    opt = torch.optim.AdamW(model.parameters(), lr=1.0)
    sched = create_scheduler(opt, t_config)

    assert sched.get_last_lr()[0] == 0.0
    for _ in range(10):
        opt.step()
        sched.step()

    assert sched.get_last_lr()[0] == 1.0  # Max LR at end of warmup
    for _ in range(90):
        opt.step()
        sched.step()

    assert sched.get_last_lr()[0] == 0.0  # 0 at max steps


def test_config_save_load():
    with tempfile.TemporaryDirectory() as d:
        path = Path(d) / "config.json"
        config = TrainingConfig(batch_size=32)
        config.save(path)

        loaded = TrainingConfig.load(path)
        assert loaded.batch_size == 32
