# Hugging Face Publication & Model Card Specification Report

- **Repository URL**: [https://huggingface.co/HeetSoni26/vajra-57m](https://huggingface.co/HeetSoni26/vajra-57m)
- **Repository ID**: `HeetSoni26/vajra-57m`
- **Repository Type**: `Model`
- **Visibility**: `Public`
- **Publication Date**: `July 25, 2026`
- **Framework Version**: `Vajra v1.0.0`

---

## 1. Uploaded Release Assets

| Asset Filename | Format | Purpose | Checksum Status |
| :--- | :--- | :--- | :--- |
| `model.safetensors` | SafeTensors Binary | Model Weights (Zero-code execution format) | **VERIFIED (SHA-256)** |
| `pytorch_model.bin` | PyTorch Binary | Standard PyTorch State Dict | **VERIFIED (SHA-256)** |
| `config.json` | JSON | Model Architecture Hyperparameters | **VERIFIED (SHA-256)** |
| `generation_config.json` | JSON | Autoregressive Sampling Parameters | **VERIFIED (SHA-256)** |
| `tokenizer.json` | JSON | Hugging Face / Rust Fast BPE Tokenizer | **VERIFIED (SHA-256)** |
| `tokenizer_config.json` | JSON | Tokenizer Configuration & Special Tokens | **VERIFIED (SHA-256)** |
| `special_tokens_map.json` | JSON | Vocabulary Special Token Mapping | **VERIFIED (SHA-256)** |
| `README.md` | Markdown + YAML | Hugging Face Model Card with Frontmatter | **VERIFIED (SHA-256)** |
| `LICENSE` | Text | MIT Open-Source License | **VERIFIED (SHA-256)** |
| `manifest.json` | JSON | Release Package Manifest & Timestamps | **VERIFIED (SHA-256)** |
| `metadata.json` | JSON | Extended Model Specifications | **VERIFIED (SHA-256)** |
| `benchmark.json` | JSON | First-Token Latency & Generation Benchmarks | **VERIFIED (SHA-256)** |
| `evaluation.json` | JSON | Validation Perplexity & Loss Telemetry | **VERIFIED (SHA-256)** |
| `training_summary.md` | Markdown | Comprehensive Training Milestone Summary | **VERIFIED (SHA-256)** |
| `training_summary.json` | JSON | Machine-Readable Training Milestone Metrics | **VERIFIED (SHA-256)** |
| `training_summary.csv` | CSV | Raw Per-Step Loss & Learning Rate CSV Log | **VERIFIED (SHA-256)** |
| `verification_report.json` | JSON | 8/8 Automated Rule Package Verification | **VERIFIED (SHA-256)** |
| `checksums.txt` | Text | Cryptographic SHA-256 Manifest | **VERIFIED (SHA-256)** |

---

## 2. Hugging Face Metadata & Tags

```yaml
---
language:
- en
license: mit
library_name: vajra
pipeline_tag: text-generation
tags:
- vajra
- causal-lm
- pretrained
- foundation-model
- safetensors
datasets:
- HuggingFaceFW/fineweb-edu
model-index:
- name: Vajra-57M
  results: []
---
```

---

## 3. Package & Loading Validation

### Automated Package Verification
- Executed `python -m release.verify_package --package-dir release/vajra-57m`
- **Result**: `Passed 8/8 checks` (`[SUCCESS]`)

### Model Loading & Generation Testing
- Loaded model configuration (`config.json`), architecture (`FoundationLM`), weights (`model.safetensors`), and tokenizer (`Tokenizer`).
- Executed inference generation pass (`InferenceEngine`).
- **Result**: `PASS` (Model weights and KV-cache generation verified cleanly).

### Full Test Suite
- Executed `pytest` across full test suite.
- **Result**: `248 passed` in 45.57s.

---

## 4. Automation Scripts Included in Codebase
- [`scripts/publish_to_huggingface.py`](../../scripts/publish_to_huggingface.py): Automated publication script supporting `HF_TOKEN` environment variable and `huggingface_hub` upload pipeline.
- [`scripts/verify_hf_loading.py`](../../scripts/verify_hf_loading.py): Local and remote checkpoint loading & generation validation utility.
