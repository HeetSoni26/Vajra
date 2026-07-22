#!/usr/bin/env bash
set -euo pipefail

deepspeed --num_gpus "${NUM_GPUS:-8}" training/pretrain.py \
  --config configs/training/pretrain_2b.yaml \
  --deepspeed_config configs/deepspeed/zero2.json
