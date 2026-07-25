"""Hugging Face Model Card (README.md) Generator for Vajra Models."""

import json
from pathlib import Path
from typing import Any


class ModelCardGenerator:
    """Generates comprehensive, Hugging Face compatible Model Cards (README.md)."""

    def __init__(self, output_dir: str | Path):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def generate(
        self,
        model_name: str,
        config: dict[str, Any],
        eval_metrics: dict[str, Any] | None = None,
        benchmark_metrics: dict[str, Any] | None = None,
        metadata: dict[str, Any] | None = None,
        training_info: dict[str, Any] | None = None,
        sample_text: str | None = None,
    ) -> Path:
        """Generate README.md file."""
        metadata = metadata or {}
        eval_metrics = eval_metrics or {}
        benchmark_metrics = benchmark_metrics or {}
        training_info = training_info or {}

        param_count = metadata.get("parameter_count", config.get("parameter_count", "N/A"))
        if isinstance(param_count, int):
            param_str = f"{param_count:,}"
        else:
            param_str = str(param_count)

        hidden_size = config.get("hidden_size", config.get("d_model", "N/A"))
        num_layers = config.get("num_layers", config.get("n_layer", "N/A"))
        num_heads = config.get("num_attention_heads", config.get("n_head", "N/A"))
        max_pos = config.get("max_position_embeddings", config.get("context_length", 2048))
        vocab_size = config.get("vocab_size", 32000)

        # YAML Frontmatter
        frontmatter = f"""---
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
- name: {model_name}
  results: []
---
"""

        # Header & Overview
        content = frontmatter + f"""
<p align="center">
  <img src="https://raw.githubusercontent.com/HeetSoni26/Vajra/main/branding/social-banner.png" alt="Vajra Banner" width="100%">
</p>

# {model_name}

## Overview
**{model_name}** is an open-weights foundation language model built using the **Vajra Framework**. 
It is designed for efficient autoregressive sequence modeling, optimized for both research and production inference.

- **Developer**: Vajra AI Team
- **Model Type**: Transformer-based Causal Language Model
- **Parameters**: {param_str}
- **Language**: English
- **License**: Apache License 2.0

---

## Architecture
{model_name} utilizes a modern decoder-only Transformer architecture with the following specifications:

- **Hidden Size ($d_{{model}}$)**: `{hidden_size}`
- **Number of Layers**: `{num_layers}`
- **Attention Heads**: `{num_heads}`
- **Context Window Length**: `{max_pos}` tokens
- **Vocabulary Size**: `{vocab_size}`
- **Position Embedding**: Rotary Position Embeddings (RoPE) / Absolute Learned
- **Normalization**: LayerNorm / RMSNorm
- **Activation Function**: SwiGLU / GELU

---

## Training Details

### Dataset
- **Pretraining Dataset**: FineWeb-Edu Sharded Corpus (High-quality educational & web tokens)
- **Token Processing**: Byte-Pair Encoding (BPE)
- **Tokens Seen**: `{metadata.get('tokens_seen', training_info.get('tokens_seen', 'N/A'))}`

### Recipe & Hardware
- **Framework**: Vajra Training Subsystem
- **Optimizer**: AdamW ($\\\\beta_1 = 0.9, \\\\beta_2 = 0.95$)
- **Precision**: Mixed Precision (BF16 / FP32)
- **Global Step**: `{metadata.get('checkpoint_step', 'N/A')}`

---

## Evaluation Metrics

| Metric | Value |
| :--- | :--- |
| **Validation Loss** | `{eval_metrics.get('validation_loss', 'N/A')}` |
| **Perplexity** | `{eval_metrics.get('perplexity', 'N/A')}` |
| **Evaluated Step** | `{eval_metrics.get('global_step', 'N/A')}` |
| **Evaluated Dataset** | `{eval_metrics.get('dataset_name', 'N/A')}` |

---

## Benchmark Performance

| Benchmark Metric | Value |
| :--- | :--- |
| **First Token Latency** | `{benchmark_metrics.get('inference_latency_first_token_ms', 'N/A')} ms` |
| **Throughput (Tokens/sec)** | `{benchmark_metrics.get('tokens_per_sec', 'N/A')}` |
| **Model Weight Size** | `{benchmark_metrics.get('model_size_mb', 'N/A')} MB` |
| **Distinct-1 Diversity** | `{benchmark_metrics.get('distinct_1', 'N/A')}` |
| **Distinct-2 Diversity** | `{benchmark_metrics.get('distinct_2', 'N/A')}` |
| **Repetition Rate** | `{benchmark_metrics.get('repetition_rate', 'N/A')}` |

---

## Example Generation

```text
{sample_text or benchmark_metrics.get('generated_sample', 'The future of AI is promising and rapidly evolving across industries.')}
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
- **Git Commit Hash**: `{metadata.get('git_commit_hash', 'N/A')}`
- **Package Version**: `{metadata.get('package_version', '1.0.0')}`
- **Packaging Timestamp**: `{metadata.get('packaging_timestamp', 'N/A')}`

---

## License
This model and its weights are released under the [Apache License 2.0](LICENSE).

---

## Citation
If you use {model_name} in your research or applications, please cite:

```bibtex
@misc{{vajra2026{model_name.lower().replace('-', '_')}}},
  author = {{Vajra AI Team}},
  title = {{{model_name}: A Scalable Foundation Language Model}},
  year = {{2026}},
  publisher = {{GitHub / Hugging Face}},
  url = {{https://github.com/HeetSoni26/Vajra}}
}}
```

---

## Version History
- **v1.0.0**: Initial public release of {model_name} base weights and release package.
"""

        path = self.output_dir / "README.md"
        with path.open("w", encoding="utf-8", newline="\n") as f:
            f.write(content)
        return path


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Create HF Model Card (README.md).")
    parser.add_argument("--model-name", default="Vajra-57M")
    parser.add_argument("--config", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()

    config = json.loads(Path(args.config).read_text(encoding="utf-8"))
    generator = ModelCardGenerator(args.output_dir)
    generator.generate(args.model_name, config)
    print(f"Generated model card at {Path(args.output_dir) / 'README.md'}")


if __name__ == "__main__":
    main()
