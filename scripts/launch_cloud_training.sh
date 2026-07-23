#!/bin/bash
# Vajra Launch Automation Script
# Usage: ./launch_cloud_training.sh [tiny|125m|370m] [--resume]

set -e

MODEL_SIZE=${1:-tiny}
RESUME_FLAG=$2

echo "=================================================="
echo "Vajra Cloud Training Orchestrator"
echo "Target Model: Vajra-${MODEL_SIZE^^}"
echo "=================================================="

# 1. Environment Verification
if ! command -v python3 &> /dev/null; then
    echo "Python3 could not be found. Please install Python 3.10+"
    exit 1
fi

echo "[1/4] Checking dependencies..."
python3 -c "import torch, transformers, wandb" || { echo "Dependencies missing. Run: pip install -r requirements.txt"; exit 1; }

# 2. Dataset Verification
echo "[2/4] Verifying dataset..."
if [ ! -f "data/tokenized/train.bin" ]; then
    echo "WARNING: Tokenized dataset not found at data/tokenized/train.bin"
    echo "Please download data and run scripts/prepare_dataset.py"
    exit 1
fi

# 3. Hardware Profiling
echo "[3/4] Profiling hardware..."
if command -v nvidia-smi &> /dev/null; then
    NUM_GPUS=$(nvidia-smi -L | wc -l)
    echo "Detected $NUM_GPUS NVIDIA GPU(s)."
else
    NUM_GPUS=0
    echo "WARNING: No NVIDIA GPUs detected."
fi

# 4. Launch Training
echo "[4/4] Launching training..."
CONFIG_FILE="configs/training/pretrain_${MODEL_SIZE}.yaml"

if [ ! -f "$CONFIG_FILE" ]; then
    echo "Config file $CONFIG_FILE does not exist!"
    exit 1
fi

if [ "$NUM_GPUS" -gt 1 ]; then
    echo "Launching via torchrun (DDP)..."
    torchrun --nproc_per_node=$NUM_GPUS -m training.pretrain --config "$CONFIG_FILE" $RESUME_FLAG
else
    echo "Launching via standard python..."
    python3 -m training.pretrain --config "$CONFIG_FILE" $RESUME_FLAG
fi

echo "Training process exited."
