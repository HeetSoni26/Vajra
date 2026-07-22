# Phase 4 — Pretraining

## Included files

- `configs/training/pretrain_1b.yaml`
- `configs/training/pretrain_2b.yaml`
- `configs/deepspeed/zero2.json`
- `configs/deepspeed/zero3.json`
- `training/pretrain.py`
- `training/trainer.py`
- `training/checkpoint.py`
- `training/launch/*`

## Operating procedure

1. Validate the tokenizer and tokenized dataset.
2. Run the 125M debug training config.
3. Confirm loss decreases and checkpoints restore.
4. Launch 1B or 2B pretraining with DeepSpeed ZeRO-2.
5. Evaluate every major token milestone.
