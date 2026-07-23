# Vajra-Tiny Training Launch Manifest

## Status: PREPARATION COMPLETE | TRAINING PENDING

This document outlines the exact requirements and commands to train the Vajra-Tiny model. Real training was **NOT** performed in this workspace because the environment lacks the necessary compute (GPU) and dataset resources (1B tokens). No model weights, checkpoints, or benchmark scores have been fabricated.

---

## 1. Hardware Requirements (Target: Vajra-Tiny)
To train Vajra-Tiny (~18M parameters) on ~1 Billion tokens in a reasonable timeframe, you need:
- **Minimum Compute**: 1x NVIDIA RTX 3060 (12GB) or RTX 4060.
- **Recommended Compute**: 1x NVIDIA RTX A4000 (16GB) or RTX 3090 / 4090 (24GB).
- **RAM**: 16GB+ System RAM.
- **Storage**: ~10GB NVMe SSD space (for tokenized dataset and checkpoints).

---

## 2. Dataset Preparation
The production tokenizer (`tokenizer/v1.0`) is validated and ready with a 65536 token vocabulary.
Before launching training, you must obtain a dataset (e.g., a subset of FineWeb, Wikipedia, or a synthetic corpus) and run the preparation script to tokenize and pack the data into `.bin` files.

**Command to Prepare Dataset:**
```bash
python -m scripts.prepare_dataset --config configs/training/pretrain_tiny.yaml
```
*Expected Output*: `data/tokenized/train.bin`, `data/tokenized/val.bin`, and a dataset manifest.

---

## 3. Training Execution
Once the dataset is prepared and GPU hardware is provisioned, launch the training script.

**Command to Launch Training (Single GPU):**
```bash
# Optional: Export your Weights & Biases API key for cloud telemetry
export WANDB_API_KEY="your_api_key_here"

# Launch the trainer
python -m training.pretrain --config configs/training/pretrain_tiny.yaml
```

**Checkpoint Layout:**
As training progresses, checkpoints will be emitted to `checkpoints/pretrain_tiny/`:
- `checkpoint_step_N.pt`: Contains model weights, optimizer state, and RNG state.
- `checkpoint_step_N.meta.json`: Sidecar file with metrics and metadata (readable without PyTorch).
- `latest.pt` / `latest.meta.json`: Symlink/copy of the most recent step for automatic resumption.
- `best.pt` / `best.meta.json`: The checkpoint with the lowest validation loss.
- `checkpoints.json`: A rolling registry of all saved checkpoints.

**Failure Recovery (Resuming):**
If the run is interrupted (e.g., OOM, node reboot), append `--resume`:
```bash
python -m training.pretrain --config configs/training/pretrain_tiny.yaml --resume
```

---

## 4. Export & Inference
When training reaches the `max_steps` defined in the config (2000 steps), the model is ready for export to standard Hugging Face formats (Safetensors + config.json).

**Command to Export:**
```bash
python -m scripts.export_hf --checkpoint checkpoints/pretrain_tiny/best.pt --output hf_vajra_tiny/
```

**Command to Validate Inference:**
```bash
python -m inference.generate --model_dir hf_vajra_tiny/ --prompt "The future of artificial intelligence is" --max_new_tokens 50
```

---

**Proceed to Phase 9 only after a real training run has been executed on a GPU instance.**
