# Vajra Distributed Training (DDP)

## Overview

The DDP extension wraps the existing single-GPU `TrainingEngine` in PyTorch
`DistributedDataParallel` enabling efficient multi-GPU training on a single node.
All previous APIs remain unchanged; DDP is purely additive.

## Architecture

```
torchrun / mp.spawn
      │
      ├── Rank 0 ──┐
      ├── Rank 1   ├── init_process_group (NCCL or Gloo)
      └── Rank N ──┘
                   │
          DDPTrainingEngine
                   │
         ┌─────────┴──────────┐
         │                    │
   wrap_model_ddp      create_distributed_dataloader
   (DDP forward)       (DistributedSampler per-rank)
         │                    │
   aggregate_metrics   rank-0 logs / checkpoints
```

## Launch

### Via `mp.spawn` (programmatic):
```python
import torch.multiprocessing as mp
from training.ddp.scripts.launch import _worker
mp.spawn(_worker, args=(world_size, train_config, ddp_config), nprocs=world_size)
```

### Via `torchrun` (recommended for production):
```bash
torchrun --nproc_per_node=4 training/ddp/scripts/launch.py train-ddp \
    --dataset-dir output/shards \
    --output-dir output/training_ddp \
    --mixed-precision bf16
```

### Via CLI helper:
```bash
python training/ddp/scripts/launch.py train-ddp \
    --dataset-dir output/shards \
    --output-dir output/training_ddp \
    --num-gpus 4
```

## Configuration

Extend `TrainingConfig` with a `DDPConfig` companion:

| Field | Default | Description |
|---|---|---|
| `enabled` | `False` | Enable DDP mode |
| `backend` | `"nccl"` | `nccl` for GPU, `gloo` for CPU/debug |
| `master_addr` | `"127.0.0.1"` | Rendezvous address |
| `master_port` | `29500` | Rendezvous port |
| `find_unused_parameters` | `False` | DDP gradient sync option |
| `static_graph` | `False` | Enables DDP static-graph optimisation |
| `timeout_minutes` | `30` | Process-group init timeout |

## Checkpointing

Only rank 0 writes checkpoints. The underlying `TrainingCheckpointManager` is
called on `unwrap_model(ddp_model)` so checkpoints are always single-GPU
compatible — they can be loaded by the standard `CheckpointManager` without
any DDP wrapping.

## Metric Aggregation

`aggregate_metrics()` uses `all_reduce` (sum + divide) to compute global
averages of `loss`, `grad_norm`, and throughput across all ranks.
A `global_tokens_per_sec` key is also added reflecting the summed throughput.

## Limitations

- Single-node only (no multi-node NCCL rendezvous via `c10d`).
- No FSDP / DeepSpeed integration.
- `_ShardMapDataset` materialises all sequences in RAM for `DistributedSampler`
  compatibility. For very large corpora, a chunked streaming approach is recommended.
