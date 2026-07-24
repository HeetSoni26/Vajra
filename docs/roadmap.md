# Vajra Project Roadmap

[Overview](../README.md) | [Architecture](architecture.md) | [Release Notes](../RELEASE_NOTES.md)

---

## Strategic Product Vision

Vajra aims to provide the most reliable, reproducible, and verifiable foundation language model stack for researchers and enterprise ML developers.

---

## Framework Milestones

### Milestone 1: Core Framework Architecture (Completed - v1.0.0)
- [x] Decoder-only LLaMA-style Transformer architecture (`RoPE`, `SwiGLU`, `RMSNorm`, `GQA`).
- [x] Custom BPE Tokenizer (`vocab_size=65536`).
- [x] Dataset sharding, streaming mixtures, and memory-mapped binary token loaders.
- [x] DDP distributed training engine with cosine annealing and gradient accumulation.
- [x] Evaluation framework (Perplexity, Loss, Diversity, Latency).
- [x] Production Release Subsystem (`release.package_model` & `release.verify_package`).
- [x] Cross-platform LF line ending normalization and static Git commit timestamping.
- [x] PyTest validation suite (248 tests passing).

---

## Model Scaling Roadmap

| Model Name | Parameters | Target Pretraining Tokens | Status |
| :--- | :--- | :--- | :--- |
| **Vajra-57M** | `90M` total / `57M` non-embed | 1.0B tokens | **Released (v1.0.0)** |
| **Vajra-125M** | `125M` | 10.0B tokens | In Progress |
| **Vajra-370M** | `370M` | 50.0B tokens | Planned (v1.2.0) |
| **Vajra-1B** | `1.1B` | 200.0B tokens | Planned (v1.5.0) |
| **Vajra-3B** | `3.2B` | 1.0T tokens | Research Phase |

---

## Research & Ecosystem Goals

- [ ] **Direct Preference Optimization (DPO)** & RLHF alignment engine.
- [ ] **Mixture-of-Experts (MoE)** sparse layer variants.
- [ ] **Long-Context Extension**: YaRN / Dynamic RoPE scaling for 32k context windows.
- [ ] **Hugging Face Model Hub**: Automated hub push script integration (`huggingface_hub`).
- [ ] **Interactive Playground**: WebUI and REST API streaming chat server.
