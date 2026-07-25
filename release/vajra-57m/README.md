---
language:
- en
license: apache-2.0
library_name: vajra
pipeline_tag: text-generation
tags:
- vajra
- causal-lm
- pretrained
- foundation-model
model-index:
- name: vajra-57m
  results: []
---

# vajra-57m

## Overview
**vajra-57m** is an open-weights foundation language model built using the **Vajra Framework**. 
It is designed for efficient autoregressive sequence modeling, optimized for both research and production inference.

- **Developer**: Vajra AI Team
- **Model Type**: Transformer-based Causal Language Model
- **Parameters**: 90,317,312
- **Language**: English
- **License**: Apache License 2.0

---

## Architecture
vajra-57m utilizes a modern decoder-only Transformer architecture with the following specifications:

- **Hidden Size ($d_{model}$)**: `512`
- **Number of Layers**: `8`
- **Attention Heads**: `8`
- **Context Window Length**: `2048` tokens
- **Vocabulary Size**: `65536`
- **Position Embedding**: Rotary Position Embeddings (RoPE) / Absolute Learned
- **Normalization**: LayerNorm / RMSNorm
- **Activation Function**: SwiGLU / GELU

---

## Training Details

### Dataset
- **Pretraining Dataset**: FineWeb-Edu Sharded Corpus (High-quality educational & web tokens)
- **Token Processing**: Byte-Pair Encoding (BPE)
- **Tokens Seen**: `64000`

### Recipe & Hardware
- **Framework**: Vajra Training Subsystem
- **Optimizer**: AdamW ($\\beta_1 = 0.9, \\beta_2 = 0.95$)
- **Precision**: Mixed Precision (BF16 / FP32)
- **Global Step**: `250`

---

## Evaluation Metrics

| Metric | Value |
| :--- | :--- |
| **Validation Loss** | `41.895` |
| **Perplexity** | `1.5659118813707052e+18` |
| **Evaluated Step** | `250` |
| **Evaluated Dataset** | `production` |

---

## Benchmark Performance

| Benchmark Metric | Value |
| :--- | :--- |
| **First Token Latency** | `72.64 ms` |
| **Throughput (Tokens/sec)** | `48.96` |
| **Model Weight Size** | `108.27 MB` |
| **Distinct-1 Diversity** | `0.0` |
| **Distinct-2 Diversity** | `0.0` |
| **Repetition Rate** | `0.0` |

---

## Example Generation

```text

```

---

## Intended Use & Limitations

### Intended Use
- Research on language model pretraining dynamics.
- Downstream task fine-tuning (e.g., instruction tuning, RAG).
- On-device and edge deployment text generation.

### Limitations & Known Issues
- **Unaligned Base Model**: This model has not undergone RLHF or instruction tuning. It will complete prompts auto-regressively without safety guardrails.
- **Hallucination**: Like all generative language models, it may produce factually incorrect statements.
- **Biases**: Base outputs reflect patterns present in public web datasets.

---

## Reproducibility Information
- **Git Commit Hash**: `06ac533`
- **Package Version**: `1.0.0`
- **Packaging Timestamp**: `2026-01-01T00:00:00Z`

---

## License
This model and its weights are released under the [Apache License 2.0](LICENSE).

---

## Citation
If you use vajra-57m in your research or applications, please cite:

```bibtex
@misc{vajra2026vajra_57m},
  author = {Vajra AI Team},
  title = {vajra-57m: A Scalable Foundation Language Model},
  year = {2026},
  publisher = {GitHub / Hugging Face},
  url = {https://github.com/HeetSoni26/Vajra}
}
```

---

## Version History
- **v1.0.0**: Initial public release of vajra-57m base weights and release package.
