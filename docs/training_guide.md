# Vajra Training Guide

This guide covers the end-to-end process of training a Vajra model.

## 1. Prepare the Dataset
Before training, you must prepare a tokenized dataset.
```bash
python -m scripts.prepare_dataset --config configs/training/pretrain_tiny.yaml
```
Ensure that `data/tokenized/train.bin` and `val.bin` exist.

## 2. Configure Training
Review the `configs/training/` YAML files. 
- Adjust `micro_batch_size` based on your VRAM.
- Adjust `gradient_accumulation_steps` to hit the desired `global_batch_tokens`.

## 3. Set up Logging
Vajra supports local JSONL/CSV, TensorBoard, and Weights & Biases (W&B).
To enable W&B:
```bash
export WANDB_API_KEY="your_key"
```

## 4. Launch Training
Run the launch commands documented in `docs/launch_commands.md`.
Use `torchrun` for multi-GPU setups.

## 5. Checkpoints
Checkpoints are saved to `checkpoints/<run_name>/`.
- `latest.pt`: The most recent step.
- `best.pt`: The step with the lowest validation loss.
- `checkpoint_step_N.pt`: History checkpoints.
Metadata is stored in `.meta.json` sidecars and `checkpoints.json`.

## 6. Failure Recovery
- **OOM**: Lower `micro_batch_size`, increase `gradient_accumulation_steps`.
- **Interruption**: Run with `--resume` to continue exactly where you left off.
- **Disk Full**: The checkpoint manager guards against disk full errors by requiring 2GB free space.

## 7. Exporting to HuggingFace
After training, convert the checkpoint for inference:
```bash
python -m scripts.export_hf --checkpoint checkpoints/pretrain_tiny/best.pt --output hf_model/
```
