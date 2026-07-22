# Distributed Scaling & Performance Optimization Guide

This document describes the multi-GPU Distributed Data Parallel (DDP) architecture, PyTorch `torchrun` invocation, Mixed Precision (AMP BF16/FP16/FP32), Gradient Checkpointing memory optimizations, and benchmarking utilities in `vajra-lm`.

## Architecture Overview

`vajra-lm` scales training across multi-GPU setups using PyTorch Distributed Data Parallel (DDP), DistributedSampler, and Automatic Mixed Precision (AMP).

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                          torchrun launcher                                  │
├──────────────────────────────────────┬──────────────────────────────────────┤
│               Rank 0                 │               Rank 1                 │
│  ┌────────────────────────────────┐  │  ┌────────────────────────────────┐  │
│  │ DistributedSampler (Split 0)   │  │  │ DistributedSampler (Split 1)   │  │
│  └────────────────────────────────┘  │  └────────────────────────────────┘  │
│                 │                    │                 │                    │
│                 ▼                    │                 ▼                    │
│  ┌────────────────────────────────┐  │  ┌────────────────────────────────┐  │
│  │ DDP Model (AMP BF16/FP16)      │  │  │ DDP Model (AMP BF16/FP16)      │  │
│  └────────────────────────────────┘  │  └────────────────────────────────┘  │
│                 │                    │                 │                    │
│                 ▼                    │                 ▼                    │
│  ┌────────────────────────────────┐  │  ┌────────────────────────────────┐  │
│  │ All-Reduce Gradients / Barrier │ <===> │ All-Reduce Gradients / Barrier │  │
│  └────────────────────────────────┘  │  └────────────────────────────────┘  │
│                 │                    │                                      │
│                 ▼                    │                                      │
│  ┌────────────────────────────────┐  │                                      │
│  │ Rank 0 Checkpoint & Metadata   │  │                                      │
│  └────────────────────────────────┘  │                                      │
└──────────────────────────────────────┴──────────────────────────────────────┘
```

## Key Distributed & Scaling Features

### 1. PyTorch DDP & `torchrun` (`training/pretrain.py`)
- **Environment Discovery**: Reads `WORLD_SIZE`, `RANK`, and `LOCAL_RANK` from environment variables populated by `torchrun`.
- **Backend**: Uses `nccl` backend on CUDA hosts, and falls back gracefully to `gloo` on CPU-only hosts.
- **Process Group Lifecycle**: Enclosed in `try ... finally:` block calling `torch.distributed.destroy_process_group()` safely on completion or failure.
- **Rank-Aware Logging**: Restricts console progress logs, model summaries, and checkpoint creation (`latest.pt`, `best.pt`) strictly to `rank 0`.
- **Seed Synchronization**: Sets random seeds dynamically per rank: `set_seed(args.seed + rank)`.

### 2. Mixed Precision AMP (`training/trainer.py`)
- **Supported Modes**: `bf16`, `fp16`, and `fp32`.
- **Hardware Autodetect**: Checks `torch.cuda.is_bf16_supported()` before enabling `bf16`. If unsupported, falls back to `fp32` safely.
- **GradScaler**: Enables `torch.amp.GradScaler("cuda")` for `fp16` mode to prevent numerical underflow. Automatically disables scaler for `bf16` and `fp32`.

### 3. Gradient Checkpointing (`model/model.py`)
- **Activation Recomputation**: Enables `torch.utils.checkpoint.checkpoint(layer, x, cos, sin, use_reentrant=False)` when `use_gradient_checkpointing=True` during training.
- **Memory Tradeoff**: Reduces VRAM activation memory by ~60–70% during deep 1B/2B model training at a minor (~15–20%) compute re-evaluation cost.

### 4. Performance Profiler & Benchmarking (`training/profiler.py`)
- Measures step time (ms), dataloader time (ms), forward time (ms), backward time (ms), optimizer time (ms), tokens/sec, samples/sec, peak VRAM, peak RAM, and scaling efficiency.
- Generates `benchmark_report.json` after benchmarking runs.

## Launch Instructions

### Multi-GPU Launch (Linux / Bash)
```bash
# Launch on 2 GPUs via torchrun
NUM_GPUS=2 bash training/launch/launch_torchrun_ddp.sh
```

### Multi-GPU Launch (Windows Batch)
```cmd
training\launch\launch_torchrun_ddp.bat
```

### Performance Benchmarking
```bash
python training/profiler.py --config configs/training/pretrain_tiny.yaml --precision fp32
```
