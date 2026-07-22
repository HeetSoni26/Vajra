#!/usr/bin/env bash
set -euo pipefail
python -m lm_eval --model hf --model_args pretrained="${MODEL_PATH:-checkpoints/final/hf}" --tasks hellaswag,piqa --batch_size 8
