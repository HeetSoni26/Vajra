@echo off
REM Windows Torchrun DDP Launcher Script
set NUM_GPUS=2
torchrun --standalone --nproc_per_node=%NUM_GPUS% training/pretrain.py --config configs/training/pretrain_1b.yaml
