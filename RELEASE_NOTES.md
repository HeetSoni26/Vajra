# Vajra v1.0.0 Release Notes

We are thrilled to announce the official **v1.0.0** public release of **Vajra**, an open-source, production-grade foundation language model framework engineered for training, evaluating, and packaging decoder-only Transformer models from scratch.

---

## Release Highlights

- **Complete Transformer Engine**: Decoder-only LLaMA-style architecture implementing Rotary Position Embeddings (RoPE), SwiGLU activations, RMSNorm, Grouped-Query Attention (GQA), and tied embeddings.
- **Vajra-57M Foundation Model**: Pretrained base model (`90M` parameters) validated on FineWeb-Edu corpora.
- **Verifiable Release Pipeline**: Production packager (`release.package_model`) and verification engine (`release.verify_package`) enforcing an audit-ready **8/8 check rule**.
- **Deterministic & Reproducible Builds**: Strict LF newline normalization and Git commit timestamp provenance ensuring bit-for-byte build parity across Linux, macOS, and Windows.
- **Agentic Memory Infrastructure**: Integrated multi-agent orchestrator, RAG context builder, persistent conversation memory, and knowledge graph storage (`vajra_agent/`).
- **Comprehensive Test Coverage**: **248 unit and integration tests** passing with 100% stability.

---

## What's New in v1.0.0

### 1. Release Packaging & Verification Subsystem (`release/`)
- Export dual format binaries (`model.safetensors` and `pytorch_model.bin`).
- Automated model card (`README.md`) generation with Hugging Face YAML metadata frontmatter.
- Automated training report generation in JSON, CSV, and Markdown (`training_summary.*`).
- Cryptographic SHA-256 manifest validation via `checksums.txt`.

### 2. Multi-GPU Distributed Training (`training/`)
- PyTorch DDP integration supporting multi-node and multi-GPU training execution.
- Automated gradient accumulation, cosine learning rate scheduler with warmup, and crash recovery.

### 3. Benchmarking & Telemetry (`benchmarks/`, `evaluation/`)
- Zero-shot evaluation metrics for validation loss, perplexity, and distinct-N n-gram diversity.
- First-token latency (TTFT) and generation throughput (tokens/second) profiling.

---

## Installation & Getting Started

```bash
git clone https://github.com/HeetSoni26/Vajra.git
cd Vajra
git lfs install
pip install -r requirements.txt
pip install -e .
```

To run package verification on the included `Vajra-57M` release:
```bash
python -m release.verify_package --package-dir release/vajra-57m
```

---

## Known Limitations & Future Plans
- **Base Model Alignment**: `Vajra-57M` is an unaligned base foundation model without RLHF or instruction tuning safety filters.
- **Future Scale**: Larger model scale weights (`Vajra-125M`, `Vajra-370M`) and Direct Preference Optimization (DPO) pipelines are planned for upcoming releases.
