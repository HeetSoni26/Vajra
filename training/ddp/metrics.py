import torch
import torch.distributed as dist
from typing import Dict, Any
from training.ddp.init import get_world_size


def all_reduce_mean(value: float) -> float:
    """
    Average a scalar across all ranks using all-reduce.
    Returns the global average; non-main ranks also receive the result.
    """
    if not dist.is_initialized():
        return value

    tensor = torch.tensor(value, dtype=torch.float64)
    dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
    return (tensor / get_world_size()).item()


def aggregate_metrics(local_metrics: Dict[str, Any]) -> Dict[str, Any]:
    """
    Reduce per-rank metrics to global averages.
    Operates in-place and returns the aggregated dict.
    Only numeric scalar values are reduced; others are passed through unchanged.
    """
    if not dist.is_initialized():
        return local_metrics

    aggregated = {}
    for k, v in local_metrics.items():
        if isinstance(v, float):
            aggregated[k] = all_reduce_mean(v)
        elif isinstance(v, int):
            aggregated[k] = int(all_reduce_mean(float(v)))
        else:
            aggregated[k] = v

    # Effective global batch: sum tokens/sec across ranks
    if "tokens_per_sec" in aggregated:
        # all_reduce already summed; divide by world_size gives per-rank average.
        # For global throughput we want the sum, so multiply back.
        aggregated["global_tokens_per_sec"] = aggregated["tokens_per_sec"] * get_world_size()

    return aggregated
