# Vajra Training Guide

This document explains the overarching training architecture.

## 1. Components
- **Tokenizer**: Custom BPE tokenizer trained on the corpus.
- **Model**: VajraForCausalLM (370M parameters).
- **Optimization**: AdamW Fused, BF16 precision, Gradient Checkpointing.
- **DDP**: Distributed Data Parallel using `torch.distributed`.

## 2. Infrastructure
- `training/workflows/orchestrator.py`: The main loop orchestrating data feeding, loss calculation, evaluation hooks, and checkpoint saving.
- `training/production/engine.py`: Handles NaN detection, metrics tracking, and gradient accumulation.

## 3. Extending Training
If you need to change hyperparameters, edit the profiles in `scripts/recipes.py` rather than altering the core `preset.py` constants unless testing new architectures.
