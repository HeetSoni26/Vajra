#!/usr/bin/env bash
set -euo pipefail

# Distributed training launcher via torchrun
NUM_GPUS="${NUM_GPUS:-2}"

torchrun --standalone --nproc_per_node="${NUM_GPUS}" training/pretrain.py \
  --config configs/training/pretrain_1b.yaml
