# Pretraining Engine Architecture & Reproducibility Guide

This document describes the pretraining pipeline, single-node PyTorch training loop, dataloader integration, failure detection mechanisms, reproducibility tracking, metrics logging, and checkpoint resume procedures in `vajra-lm`.

## Architecture Overview

The pretraining engine connects binary dataset arrays (`train.bin`, `val.bin`) to the decoder-only Transformer model via PyTorch `DataLoader` instances and `np.memmap`.

```
┌─────────────────┐     ┌─────────────────────┐     ┌──────────────────────┐
│ Memory-Mapped   │ ──> │ PyTorch DataLoader  │ ──> │ Transformer Model    │
│ Dataset (.bin)  │     │ (Batching, Shuffle) │     │ (Forward & Loss)     │
└─────────────────┘     └─────────────────────┘     └──────────────────────┘
                                                                │
                                                                ▼
┌─────────────────┐     ┌─────────────────────┐     ┌──────────────────────┐
│ Checkpoints &   │ <── │ Early Failure       │ <── │ Optimizer & Cosine   │
│ Summaries       │     │ Checks (NaN/Inf)    │     │ LR Scheduler         │
└─────────────────┘     └─────────────────────┘     └──────────────────────┘
```

## Key Components

### 1. Dataset Dataloader (`training/data_loader.py`)
- **Memmap Access**: Reads `train.bin` and `val.bin` as `np.uint32` arrays with zero memory copy.
- **Sequence Splitting**: Generates non-overlapping input IDs and target labels (`input_ids = chunk[:-1]`, `labels = chunk[1:]`).
- **Deterministic Validation**: Training dataloader uses `shuffle=True`, while validation dataloader uses `shuffle=False` to ensure exact evaluation comparisons across steps.

### 2. Training Engine (`training/trainer.py`)
- **Gradient Accumulation**: Accumulates gradients over `gradient_accumulation_steps` before stepping optimizer weights.
- **Cosine Learning Rate**: Dynamically updates parameter learning rates per step using linear warmup and cosine decay (`training/optimizer.py`).
- **Gradient Clipping**: Enforces norm clipping via `torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)`.
- **Early Failure Detection**: Instantly aborts training if:
  - Loss value evaluates to `NaN` or `Inf`.
  - Gradient norm evaluates to `NaN` or `Inf`.
  - Gradient norm exceeds `max_grad_norm_threshold` (default: 100.0).

### 3. Model Summary & Reproducibility (`training/pretrain.py`)
- **Model Summary**: Generates `model_summary.json` before training: total parameters, trainable parameters, non-trainable parameters, estimated FLOPs/token, estimated VRAM, layers, hidden size, vocabulary size.
- **Experiment Reproducibility**: Writes `experiment_metadata.json` capturing random seed, Git commit hash, dataset manifest checksums, tokenizer version, full training configuration, and optimizer parameters.
- **Checkpoint Resume**: Integrates `CheckpointManager` from Sprint 2. Launching with `--resume` seamlessly restores model weights, optimizer states, global step, and tokens processed.
- **Training Curves & Summary**: Generates `training_summary.json` containing total training time, completed steps, tokens processed, final loss, and step history.

## Configuration Guide (`configs/training/pretrain_tiny.yaml`)

```yaml
model_config: configs/model/model_tiny.yaml
data_dir: data/tokenized
output_dir: checkpoints/pretrain_tiny
sequence_length: 128
micro_batch_size: 2
gradient_accumulation_steps: 2
learning_rate: 1.0e-3
min_learning_rate: 1.0e-4
warmup_steps: 10
weight_decay: 0.01
adam_beta1: 0.9
adam_beta2: 0.95
adam_epsilon: 1.0e-8
grad_clip: 1.0
max_steps: 50
save_every_steps: 10
eval_every_steps: 10
precision: float32
```

## Running Pretraining & Resuming

```bash
# Launch tiny pretraining smoke test
python training/pretrain.py --config configs/training/pretrain_tiny.yaml

# Resume pretraining from latest checkpoint
python training/pretrain.py --config configs/training/pretrain_tiny.yaml --resume
```
