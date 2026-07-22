#!/usr/bin/env bash
set -euo pipefail

python training/pretrain.py --config configs/training/debug_125m.yaml
