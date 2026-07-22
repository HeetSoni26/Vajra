# Section 2 — Architecture Design

## Design philosophy

The model follows the LLaMA-lineage decoder-only transformer family: RoPE positional encoding, pre-norm RMSNorm, SwiGLU feed-forward layers, grouped-query attention, causal masking, BF16 training, and optional FlashAttention. Components are selected for empirical reliability and inference compatibility rather than novelty.

## Model dimensions

| Hyperparameter | 1B variant | 2B variant | Rationale |
|---|---:|---:|---|
| Layers | 24 | 28 | balanced depth at small-model scale |
| Hidden size | 2048 | 2560 | tensor-core friendly widths |
| Attention heads | 16 | 20 | 128-dimensional heads |
| KV heads | 8 | 4 | GQA reduces KV cache |
| FFN intermediate | 5632 | 6912 | SwiGLU rounded to hardware-friendly multiples |
| Context length | 4096 | 4096 | efficient default, extendable later |
| Vocabulary size | 65536 | 65536 | code/math friendly and power-of-two |
| Weight tying | yes | yes | saves embedding-sized output head |

## Parameter count formula

For a decoder-only model with tied LM head:

```text
P_embed = vocab_size × hidden_size
P_attn  = hidden×hidden + 2×hidden×(kv_heads×head_dim) + hidden×hidden
P_ffn   = 3 × hidden × intermediate_size
P_norm  = 2 × hidden per layer plus final norm
Total   = P_embed + layers × (P_attn + P_ffn + P_norm) + final_norm
```

The checked-in `scripts/count_parameters.py` computes the implemented count from the PyTorch module and should be used as the source of truth before training.

## Component decisions

### RoPE positional encoding

RoPE is selected because it is widely used in modern open LLMs, works well with length-extension methods, and is supported by mainstream inference stacks. Learned absolute embeddings are rejected due to poor extrapolation.

### RMSNorm with pre-norm residuals

RMSNorm reduces normalization cost compared with LayerNorm and is stable in pre-norm transformer blocks. Pre-norm is required for reliable deep-transformer optimization.

### SwiGLU feed-forward network

SwiGLU uses gate, up, and down projections. It costs more than a two-matrix GeLU FFN but has better perplexity-per-parameter behavior in modern decoder-only models.

### Grouped-query attention

GQA shares K/V heads across groups of query heads. It preserves most MHA quality while reducing KV-cache memory and improving serving efficiency. The scaffold supports 8 KV heads for the 1B variant and 4 KV heads for the 2B variant.

### Weight tying

The token embedding matrix and LM head are tied by default. This saves a large vocabulary projection matrix and is appropriate for 1B–2B scale models unless an ablation proves otherwise.

### Context length

4096 tokens is the default pretraining context length because it is useful for real workloads while avoiding the full cost of longer quadratic attention. Longer context should be added after base training through RoPE scaling and continued training.

## Implementation map

| Component | File |
|---|---|
| Configuration | `model/config.py` |
| RMSNorm | `model/norm.py` |
| RoPE | `model/rope.py` |
| GQA attention | `model/attention.py` |
| SwiGLU | `model/feedforward.py` |
| Decoder model | `model/model.py` |
| Generation helper | `model/generation.py` |

## Baseline comparison

| Model | Params | Layers | Hidden | Heads | KV heads | Context | Vocab |
|---|---:|---:|---:|---:|---:|---:|---:|
| Ours 1B | ~1B | 24 | 2048 | 16 | 8 | 4096 | 65536 |
| Ours 2B | ~2B | 28 | 2560 | 20 | 4 | 4096 | 65536 |
| TinyLlama-1.1B | 1.1B | 22 | 2048 | 32 | 4 | 2048 | 32000 |
| OLMo-1B | 1.2B | 16 | 2048 | 16 | 16 | 2048 | 50280 |
| SmolLM-1.7B | 1.7B | 24 | 2048 | 32 | 8 | 2048 | 49152 |

## Validation requirements

- `tests/test_model.py` passes for a tiny config.
- `scripts/count_parameters.py` runs for both 1B and 2B configs.
- Tiny overfit experiment decreases loss.
- Checkpoint save/load preserves model and optimizer state.
- Export path to HuggingFace is tested before full-scale training.
