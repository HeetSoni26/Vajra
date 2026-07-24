# Vajra v1.0.0 Stable Release Notes

**Release Date**: July 24, 2026  
**Tag**: `v1.0.0`  
**Model Release**: `Vajra-57M` Base Foundation Model  
**License**: Apache-2.0 / MIT  
**Repository**: [https://github.com/HeetSoni26/Vajra](https://github.com/HeetSoni26/Vajra)

---

## 🚀 Summary

We are thrilled to present the official **v1.0.0 Stable Release of the Vajra Framework**, accompanying the publication of our first production model release package: **Vajra-57M**.

Vajra is an end-to-end open-source AI ecosystem integrating high-performance decoder-only transformer architecture (**Vajra-LM**), an autonomous agent system (**Vajra-Agent**), and a fully automated Dataset, Evaluation, Benchmarking, and Packaging pipeline.

> [!NOTE]  
> **Production Release Verification**:  
> The **Vajra-57M** model release package (`release/vajra-57m`) has been packaged with complete SHA-256 checksums, reproducibility manifests, Hugging Face compatible Model Cards (`README.md`), evaluation loss metrics, generation quality telemetry, and hardware throughput profiles. **Verification Status: 8/8 CHECKS PASSED.**

---

## 🔥 Release Highlights

### 1. Vajra-57M Foundation Model Package (`release/vajra-57m`)
- **SafeTensors Weights**: Fully exported `model.safetensors` compatible with Hugging Face transformers.
- **BPE Tokenizer**: Fully exportable `tokenizer.json`, `tokenizer_config.json`, and `special_tokens_map.json`.
- **Reproducibility Manifest**: `manifest.json` locking git commit hashes, parameter count (90.3M total / 56.7M backbone parameters), checkpoint steps, and token counts.
- **Checksum Security**: Verified SHA-256 hashes generated in `checksums.txt`.

### 2. Evaluation & Benchmarking Subsystems
- **Native Evaluation Framework**: Standardized evaluation (`evaluate.py`, `evaluate_all.py`) computing cross-entropy loss and perplexity.
- **Hardware & Quality Benchmarking**: Automated metrics (`benchmarks/`) measuring tokens/sec throughput, first-token latency, memory footprint, and N-gram generation diversity (Distinct-1, Distinct-2, Repetition Rate).

### 3. Core Architecture & Engineering
- **Vajra-LM Core**: SwiGLU activation, RoPE positional embeddings, RMSNorm, and Grouped-Query Attention (GQA).
- **Inference & Serving Engine**: Native streaming generator with KV-cache prefilling support and FastAPI serving integration.
- **Automated Verification**: `verify_package.py` audit runner outputting `verification_report.json`.

---

## 📦 Installation & Release Artifact Usage

```bash
# Clone repository
git clone https://github.com/HeetSoni26/Vajra.git
cd Vajra

# Install full environment
pip install -e .[all]

# Verify the official release package
python -m release.verify_package --package-dir release/vajra-57m
```

---

## 📄 License & Attribution

Licensed under the **Apache License 2.0 / MIT License**.
