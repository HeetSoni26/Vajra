# Vajra Public Product Roadmap

Welcome to the public roadmap for the **Vajra Foundation Language Model Framework**.

---

## Current Status: v1.0.0 (Released)

The framework has reached production stability with complete pretraining, evaluation, and verifiably signed release packaging capabilities.

---

## Roadmap Overview

### 1. Framework & Core Architecture
- [x] Decoder-only LLaMA-style Transformer (`RoPE`, `SwiGLU`, `RMSNorm`, `GQA`).
- [x] Custom Hugging Face BPE Tokenizer (`vocab_size=65536`).
- [x] Multi-source streaming dataset loader & binary memory-mapped sharding.
- [x] Distributed Data Parallel (DDP) pretraining engine with cosine decay.
- [x] Perplexity, Loss, Diversity, and Latency benchmarking engine.
- [x] Release packaging engine (`model.safetensors`, `pytorch_model.bin`, `manifest.json`).
- [x] Cross-platform LF line ending normalization & static Git commit timestamping.
- [x] Git LFS tracking for release weight binaries.
- [x] 8/8 rule release integrity verifier script.
- [x] Comprehensive PyTest test suite (**248 tests passing**).

### 2. Model Family Scaling
- [x] **Vajra-57M**: Base foundation model release (`90M` total params / `57M` non-embed).
- [ ] **Vajra-125M**: Small scale pretrained model (Target: 10B tokens).
- [ ] **Vajra-370M**: Intermediate scale pretrained model (Target: 50B tokens).
- [ ] **Vajra-1B**: Standard foundation scale model (Target: 200B tokens).
- [ ] **Vajra-3B**: Large scale foundation model (Target: 1T tokens).

### 3. Research & Post-Training
- [ ] Direct Preference Optimization (DPO) & SFT alignment pipeline.
- [ ] Sparse Mixture-of-Experts (MoE) architecture modules.
- [ ] Long-context RoPE extension (8k / 32k tokens).
- [ ] Multimodal vision-language integration.

### 4. Ecosystem & Infrastructure
- [ ] Direct Hugging Face Hub push workflow (`huggingface_hub` integration).
- [ ] Production REST API inference server & streaming Web UI.
- [ ] Docker & Kubernetes pretraining deployment scripts.
