import os
import torch.distributed as dist
import torch.multiprocessing as mp

from training.ddp.config import DDPConfig
from training.ddp.init import (
    get_rank,
    get_world_size,
    is_main_process,
    barrier,
)
from training.ddp.wrapper import unwrap_model
from training.ddp.metrics import aggregate_metrics, all_reduce_mean
from model.config import VajraConfig
from model.modeling import VajraForCausalLM


# ---------------------------------------------------------------------------
# Helpers for running distributed tests in a single process using Gloo
# ---------------------------------------------------------------------------


def _run_in_gloo(fn, world_size: int = 2):
    """Spawns `world_size` processes running `fn(rank, world_size)` via Gloo."""
    mp.spawn(fn, args=(world_size,), nprocs=world_size, join=True)


def _init_gloo(rank: int, world_size: int):
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = "29600"
    os.environ["USE_LIBUV"] = "0"
    dist.init_process_group(backend="gloo", rank=rank, world_size=world_size)


def _cleanup():
    if dist.is_initialized():
        dist.destroy_process_group()


# ---------------------------------------------------------------------------
# Unit tests that do NOT require distributed init
# ---------------------------------------------------------------------------


def test_ddp_config_defaults():
    cfg = DDPConfig()
    assert cfg.backend == "nccl"
    assert cfg.master_port == 29500
    assert not cfg.find_unused_parameters


def test_unwrap_plain_model():
    """unwrap_model should return the model unchanged when it is not wrapped."""
    config = VajraConfig(
        vocab_size=64,
        hidden_size=32,
        intermediate_size=64,
        num_layers=1,
        num_attention_heads=2,
        num_key_value_heads=1,
    )
    model = VajraForCausalLM(config)
    assert unwrap_model(model) is model


def test_aggregate_metrics_no_dist():
    """Without dist, aggregate_metrics is a no-op pass-through."""
    metrics = {"loss": 2.5, "tokens_per_sec": 1000.0}
    result = aggregate_metrics(metrics)
    assert result["loss"] == 2.5
    assert result["tokens_per_sec"] == 1000.0


def test_all_reduce_mean_no_dist():
    """Without dist, all_reduce_mean returns the value unchanged."""
    assert all_reduce_mean(3.14) == 3.14


# ---------------------------------------------------------------------------
# Multi-process tests using Gloo (CPU, works offline)
# ---------------------------------------------------------------------------


def _barrier_worker(rank: int, world_size: int):
    _init_gloo(rank, world_size)
    barrier()
    _cleanup()


def test_barrier_two_processes():
    _run_in_gloo(_barrier_worker, world_size=2)


def _all_reduce_worker(rank: int, world_size: int):
    _init_gloo(rank, world_size)
    # Each rank contributes its rank value; mean should be 0.5 for ranks [0,1]
    result = all_reduce_mean(float(rank))
    expected = sum(range(world_size)) / world_size
    assert abs(result - expected) < 1e-5, f"Expected {expected}, got {result}"
    _cleanup()


def test_all_reduce_mean_two_processes():
    _run_in_gloo(_all_reduce_worker, world_size=2)


def _rank_check_worker(rank: int, world_size: int):
    _init_gloo(rank, world_size)
    assert get_rank() == rank
    assert get_world_size() == world_size
    assert is_main_process() == (rank == 0)
    _cleanup()


def test_rank_discovery_two_processes():
    _run_in_gloo(_rank_check_worker, world_size=2)
