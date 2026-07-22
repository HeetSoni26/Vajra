# Model Card for FoundationLM-1B

## Model Details

- **Model Name**: FoundationLM-1B
- **Architecture**: Decoder-only Transformer (LLaMA-style, SwiGLU, RMSNorm, GQA, RoPE)
- **Parameters**: 1.1 Billion
- **Context Length**: 4,096 tokens
- **Vocabulary Size**: 65,536 tokens (Byte-Level BPE)
- **License**: Apache 2.0

## Intended Use

### Primary Uses
- Text generation, reasoning, and completion tasks.
- Downstream fine-tuning for specialized domain applications.

### Out-of-Scope Uses
- Real-time safety-critical decisions without human oversight.
- Generating harmful, abusive, or unlawful content.

## Training Data & Procedure

- **Dataset**: Curated open-access multi-domain text corpus.
- **Hardware**: Multi-GPU DDP cluster using PyTorch AMP (BF16).
- **Optimizer**: AdamW ($\beta_1=0.9, \beta_2=0.95$, weight decay 0.1) with cosine learning rate decay.

## Evaluation Results

- **Validation Perplexity**: Measured on `val.bin` memmap dataset.
- **lm-evaluation-harness**: Evaluated across primary benchmarks.
