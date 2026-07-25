# GitHub Release v1.0.0 Asset Manifest & Inventory

This document defines the complete asset manifest and distribution inventory accompanying the **Vajra v1.0.0** public GitHub Release.

---

## 1. Primary Model Package Artifacts (`release/vajra-57m/`)

| Asset Name | Format | Description |
| :--- | :--- | :--- |
| `model.safetensors` | `.safetensors` | Hugging Face compatible SafeTensors binary containing 90.3M model parameters. |
| `pytorch_model.bin` | `.bin` | PyTorch `state_dict` binary. |
| `config.json` | JSON | Hugging Face compatible model configuration file. |
| `generation_config.json` | JSON | Autoregressive sampling and token generation hyperparameters. |
| `tokenizer.json` | JSON | Fast BPE Tokenizer definition (`vocab_size=65536`). |
| `tokenizer_config.json` | JSON | Tokenizer configuration and special token IDs mapping. |
| `special_tokens_map.json` | JSON | Mapping of BOS, EOS, PAD, UNK, and control tokens. |
| `metadata.json` | JSON | Git commit hash, build version, parameter count, and provenance timestamp. |
| `manifest.json` | JSON | Full file inventory manifest with byte sizes and SHA-256 signatures. |
| `checksums.txt` | Text | Plaintext SHA-256 checksum signatures for all package files. |
| `README.md` | Markdown | Hugging Face Model Card with YAML frontmatter. |
| `LICENSE` | Text | MIT Software License. |

---

## 2. Release Verification & Telemetry Reports

| Asset Name | Format | Description |
| :--- | :--- | :--- |
| `verification_report.json` | JSON | Machine-readable report proving **8/8 PASS** verification status. |
| `training_summary.json` | JSON | Machine-readable pretraining progression summary. |
| `training_summary.csv` | CSV | Tabular metrics log for pretraining execution. |
| `training_summary.md` | Markdown | Executive markdown report for training progression. |
| `evaluation.json` | JSON | Validation Loss (`41.895`) and Perplexity metrics. |
| `benchmark.json` | JSON | First Token Latency (`72.64 ms`) and Throughput (`48.96 tokens/sec`) metrics. |

---

## 3. Documentation & Verification Assets

| Asset Name | Location | Description |
| :--- | :--- | :--- |
| **Release Page Text** | `docs/releases/v1.0.0.md` | Copy/paste release text for the GitHub Release UI. |
| **Release Notes** | `RELEASE_NOTES.md` | Executive summary of features, fixes, and telemetry. |
| **Training Completion Report** | `docs/models/vajra-57m-training-report.md` | Comprehensive engineering record for `Vajra-57M`. |
| **Project Vision** | `VISION.md` | Long-term mission, philosophy, and architectural direction. |
| **Project Roadmap** | `ROADMAP.md` | Future model family scaling and ecosystem milestones. |
| **Replication Command Guide** | `README.md#quick-start` | Verification & execution CLI commands. |
