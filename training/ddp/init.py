import datetime
import os

import torch.distributed as dist

from training.ddp.config import DDPConfig


def init_process_group(config: DDPConfig, rank: int, world_size: int):
    """
    Initializes the distributed process group.
    Must be called before any distributed operations.
    """
    os.environ["MASTER_ADDR"] = config.master_addr
    os.environ["MASTER_PORT"] = str(config.master_port)

    dist.init_process_group(
        backend=config.backend,
        rank=rank,
        world_size=world_size,
        timeout=datetime.timedelta(minutes=config.timeout_minutes),
    )


def cleanup():
    """Gracefully tears down the distributed process group."""
    if dist.is_initialized():
        dist.destroy_process_group()


def get_rank() -> int:
    if dist.is_initialized():
        return dist.get_rank()
    return 0


def get_world_size() -> int:
    if dist.is_initialized():
        return dist.get_world_size()
    return 1


def get_local_rank() -> int:
    return int(os.environ.get("LOCAL_RANK", 0))


def is_main_process() -> bool:
    return get_rank() == 0


def barrier():
    """Synchronise all ranks."""
    if dist.is_initialized():
        dist.barrier()
