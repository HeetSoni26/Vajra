# Section 5 — Training Infrastructure

## 5.1 Objective

Provide reliable, monitored, recoverable distributed training for 1B–2B parameter models using BF16, FlashAttention-compatible attention, DeepSpeed/FSDP-style sharding, and memory-mapped dataset loading.

Implementation entry points:

- `configs/training/pretrain_1b.yaml`
- `configs/training/pretrain_2b.yaml`
- `configs/training/debug_125m.yaml`
- `configs/deepspeed/zero2.json`
- `configs/deepspeed/zero3.json`
- `training/pretrain.py`
- `training/trainer.py`
- `training/checkpoint.py`
- `training/data_loader.py`
- `training/optimizer.py`

## 5.2 Hardware configurations

| Target | Minimum | Recommended | Notes |
|---|---|---|---|
| 125M debug | 1× A10/A100/H100 | 1× A100 | validates pipeline cheaply |
| 1B pretraining | 4× A100 40GB | 8× A100 80GB | ZeRO-2 usually sufficient |
| 2B pretraining | 4× A100 80GB | 8× A100/H100 80GB | use ZeRO-3 if memory pressure appears |

NVLink or equivalent high-bandwidth interconnect is recommended. Multi-node training should be added only after single-node throughput is stable.

## 5.3 Software stack

| Layer | Tooling |
|---|---|
| Framework | PyTorch 2.3+ |
| Distributed | DeepSpeed ZeRO-2 primary, ZeRO-3 fallback, FSDP optional |
| Precision | BF16 mixed precision |
| Attention | PyTorch SDPA fallback, FlashAttention path on target host |
| Tokenizer/data | HuggingFace tokenizers, NumPy memmap, PyArrow/Parquet as intermediate |
| Tracking | Weights & Biases plus local logs |
| Config | YAML configs committed to repository |

## 5.4 Hyperparameters

Default 1B settings:

| Hyperparameter | Value |
|---|---:|
| sequence length | 4096 |
| global batch tokens | 2M |
| peak LR | 3e-4 |
| min LR | 3e-5 |
| warmup | 2000 steps |
| optimizer | AdamW |
| betas | 0.9, 0.95 |
| weight decay | 0.1 |
| grad clipping | 1.0 |
| dropout | 0.0 |
| precision | BF16 |

Default 2B settings use 4M global batch tokens, peak LR 2e-4, and min LR 2e-5.

## 5.5 Compute estimate

Use the standard approximation:

```text
training FLOPs ≈ 6 × parameters × tokens
```

This should be tracked with actual model FLOP utilization during training. Keep a cost-per-billion-token dashboard and compare spot versus on-demand pricing.

## 5.6 Checkpoint strategy

Required checkpoint fields:

- model weights
- optimizer state
- scheduler state
- random number generator state
- global step
- tokens seen
- dataloader/shard position
- config snapshot

Default policy:

- save every 1000 steps
- keep last 5 rolling checkpoints
- keep permanent milestone checkpoints at major token counts
- validate checkpoint restore on the debug run before main training

## 5.7 Monitoring

Log these metrics every step or every few steps:

| Metric | Purpose |
|---|---|
| training loss | primary optimization signal |
| validation loss / perplexity | overfitting and data-quality signal |
| grad norm | divergence detection |
| learning rate | schedule validation |
| tokens seen | progress and cost accounting |
| GPU memory | memory regression detection |
| GPU utilization | dataloader bottleneck detection |
| MFU | training efficiency |

Alert on sustained loss spikes, grad norm > 10, GPU utilization below 80%, NaNs, checkpoint failures, or validation perplexity regressions.

## 5.8 Stability protocol

If loss spikes persist:

1. Stop training after confirming the spike is not logging noise.
2. Restore the most recent stable checkpoint.
3. Inspect the triggering batch or shard.
4. Add bad document fingerprints to a blocklist if needed.
5. Resume with temporarily reduced LR.
6. Document the incident in the training report.

## 5.9 Validation criteria

- Debug model trains and restores checkpoints.
- Main config launches on target hardware without OOM.
- Loss decreases on a fixed small corpus.
- W&B or local tracking records all required metrics.
- Throughput is stable and input pipeline is not the bottleneck.
