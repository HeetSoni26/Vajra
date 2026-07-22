# Phase 3 — Model Architecture

## Included components

- `ModelConfig`
- RMSNorm
- RoPE
- SwiGLU FFN
- grouped-query causal attention
- decoder block
- tied embedding/LM head support
- parameter counting script

## Validation checklist

- Parameter count matches target within tolerance.
- Forward pass returns `[batch, sequence, vocab]` logits.
- Gradients flow through every trainable parameter.
- Tiny overfit experiment decreases loss.
