# Frequently Asked Questions (FAQ)

[Overview](../README.md) | [Architecture](architecture.md) | [Release Pipeline](release_pipeline.md)

---

### Q1: What makes Vajra different from standard Hugging Face pretraining scripts?
**Answer**: Vajra is designed as an end-to-end framework emphasizing deterministic, auditably verified releases. It manages dataset sharding, tokenizer training, custom PyTorch Transformer layers, distributed training, benchmark telemetry, and release package verification (with SHA-256 signatures and Hugging Face `.safetensors` exports) natively.

### Q2: How does the release verifier work?
**Answer**: The verifier script (`python -m release.verify_package --package-dir release/vajra-57m`) checks 8 criteria: presence of required metadata files, existence of weights (`model.safetensors`/`pytorch_model.bin`), tokenizer configs, byte-for-byte SHA-256 match in `checksums.txt`, successful model weight loading, config key validation, generation config parameters, and manifest/metadata consistency.

### Q3: Why does `release.package_model` generate deterministic line endings?
**Answer**: On Windows, default file writes append CRLF (`\r\n`), whereas Linux/macOS write LF (`\n`). To guarantee cross-platform checksum reproducibility, all text, JSON, CSV, and Markdown files generated during packaging explicitly enforce LF (`\n`) and static Git commit timestamps.

### Q4: Can I load Vajra weights in Hugging Face `transformers`?
**Answer**: Yes! The release pipeline exports standard `model.safetensors` and `config.json` formatted for `FoundationLM` architecture compatible with standard Hugging Face model loading.

### Q5: How many tests are in the PyTest suite?
**Answer**: The test suite contains **248 unit and integration tests** covering architecture, tokenization, pretraining loops, DDP serialization, memory systems, and release verification.
