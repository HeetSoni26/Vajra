# Inference Engine Architecture & Usage Guide

The `InferenceEngine` module (`inference/engine.py`) provides high-throughput autoregressive decoding with Key-Value (KV) caching, multiple sampling methods, batch inference, and token streaming.

## Architecture

```
┌────────────────────────────────────────────────────────────────┐
│                      InferenceEngine                           │
├────────────────────────────────────────────────────────────────┤
│ 1. Encode prompt -> input_ids                                  │
│ 2. Initialize per-layer KVCache (prefill mode)                 │
│ 3. Forward pass through FoundationLM                           │
│ 4. Apply Repetition Penalty -> Top-K -> Top-P -> Temperature   │
│ 5. Decode step: feed only new token into KVCache               │
│ 6. Stream or return decoded text                               │
└────────────────────────────────────────────────────────────────┘
```

## Features

- **KV Cache Optimization**: Caches keys and values across layers to eliminate recomputing past activations. Achieves up to **1.24x–3.5x speedup** depending on context length.
- **Sampling Strategies**:
  - Greedy decoding (`do_sample=False` or `temperature=0.0`)
  - Temperature scaling (`temperature`)
  - Top-K truncation (`top_k`)
  - Top-P Nucleus filtering (`top_p`)
  - Repetition penalty (`repetition_penalty`)
- **Execution Modes**:
  - Single prompt generation (`engine.generate(prompt)`)
  - Batch generation (`engine.generate([p1, p2, ...])`)
  - Streaming token output (`engine.generate_stream(prompt)`)
- **Precision Support**: `fp32`, `fp16`, and `bf16` via `torch.amp.autocast`.

## Quick Start Code Example

```python
from inference.engine import InferenceEngine, GenerationConfig

# Load engine from training config & checkpoint
engine = InferenceEngine.from_config(
    config_path="configs/training/pretrain_tiny.yaml",
    checkpoint="checkpoints/run/latest.pt",
)

# Configure generation settings
gen_cfg = GenerationConfig(
    max_new_tokens=64,
    temperature=0.7,
    top_k=50,
    top_p=0.9,
    use_kv_cache=True,
)

# Single generation
text = engine.generate("Once upon a time", gen_cfg)
print(text[0])

# Streaming token-by-token
for token in engine.generate_stream("In a galaxy far away", gen_cfg):
    print(token, end="", flush=True)
```
