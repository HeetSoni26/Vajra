# Hardware Profiles for Vajra Framework

This guide provides hardware requirements and expected profiles for training Vajra models.

## Vajra-Tiny (~18M Parameters)
- **Target Context**: 2048 tokens
- **Training Tokens**: ~1B
- **Minimum VRAM**: 6 GB
- **Recommended Hardware**: 
  - 1x RTX 3060 / 4060 (Local Dev)
  - 1x RTX A4000
- **DDP**: Not required.
- **Estimated Training Time (1x RTX 4090)**: ~2-4 hours.

## Vajra-125M (~125M Parameters)
- **Target Context**: 2048 tokens
- **Training Tokens**: ~5B
- **Minimum VRAM**: 24 GB
- **Recommended Hardware**: 
  - 1x RTX 3090 / 4090
  - 1x RTX A6000 / A100 (40GB)
  - 2x RTX 4090 (DDP)
- **DDP**: Recommended for faster training.
- **Estimated Training Time (1x A100)**: ~1-2 days.

## Vajra-370M (~370M Parameters)
- **Target Context**: 4096 tokens
- **Training Tokens**: ~15B
- **Minimum VRAM**: 40 GB per GPU (with Gradient Checkpointing / Zero DP)
- **Recommended Hardware**: 
  - 4x A100 (80GB)
  - 8x RTX A6000
- **DDP**: Required.
- **Estimated Training Time (4x A100 80GB)**: ~3-5 days.

### Notes on Memory (VRAM)
VRAM requirements scale with:
1. **Model Size**: ~16-18 bytes per parameter (AdamW optimizer states + gradients + weights).
2. **Batch Size / Seq Len**: Activations scale linearly with sequence length and batch size. Use `gradient_accumulation_steps` to trade compute time for memory.
