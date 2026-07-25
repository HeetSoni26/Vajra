# Vajra v1.0.0 Release Notes

We are thrilled to present the official **v1.0.0** release of **Vajra**, an open-source, production-grade foundation language model framework engineered for transparent pretraining, automated evaluation, and auditably verified release packaging.

---

## Release Highlights

- **Vajra-57M Base Foundation Model**: Pretrained base model (`90,317,312` parameters) trained on sharded FineWeb-Edu token mixtures.
- **8/8 Verification System**: Built-in release packaging and verification pipeline enforcing 8 strict integrity gates covering SHA-256 manifest matching, weight loading, tokenizer assets, and metadata synchronization.
- **Deterministic Cross-Platform Packaging**: Strict LF (`\n`) line ending normalization and static Git commit timestamp integration ensuring byte-for-byte build parity across Linux, macOS, and Windows.
- **Stateful Agent & Memory Infrastructure**: Integrated multi-agent orchestrator, persistent conversation memory, RAG context builders, and factual knowledge graph storage (`vajra_agent/`).
- **Comprehensive Testing Parity**: **248 unit and integration tests** passing with 100% stability.

---

## Technical & Architectural Improvements

1. **Decoder-Only LLaMA Lineage Architecture**:
   - Rotary Position Embeddings (RoPE, $\theta = 10000.0$)
   - Root Mean Square Normalization (RMSNorm, $\epsilon = 1.0 \times 10^{-6}$)
   - SwiGLU Activation Function ($d_{ff} = 1376$)
   - Grouped-Query Attention (GQA, $N_q=8, N_{kv}=4$)
   - Tied Embedding Matrix (`tie_word_embeddings = true`)

2. **Distributed Training Subsystem**:
   - PyTorch DDP integration with automated gradient accumulation, cosine decay scheduler, and crash-resilient checkpointing.

3. **Benchmarking Telemetry**:
   - Evaluation metrics covering Perplexity, Validation Loss, First Token Latency (TTFT = `72.64 ms`), and Generation Throughput (`48.96 tokens/sec`).

---

## Verification & Testing Summary

- **Package Verification Status**: `SUCCESS (Passed 8/8 checks)`
- **PyTest Execution**: `248 passed` tests across architecture, tokenization, training, memory, and release modules.

---

## Known Limitations

- **Pretraining Milestone**: Checkpoint Step 250 represents an initial pretraining milestone run (64,000 tokens processed out of a target 1B token budget).
- **Unaligned Foundation Model**: `Vajra-57M` is an unaligned base model without RLHF or instruction tuning safety filters. Downstream applications should implement guardrails.

---

## Future Roadmap

- Pretraining scaling for `Vajra-125M` and `Vajra-370M`.
- Direct Preference Optimization (DPO) and SFT alignment engine.
- Sparse Mixture-of-Experts (MoE) modules.

---

## Version & License Information

- **Release Version**: `v1.0.0`
- **Git Commit Hash**: `9b4ffafecd784387446a008c40081477a3649811`
- **License**: [MIT License](LICENSE)
- **Documentation**: [Full Technical Docs Index](docs/README.md)
