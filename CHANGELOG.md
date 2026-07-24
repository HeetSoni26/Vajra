# Changelog

All notable changes to the **Vajra** project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

---

## [1.0.0] - 2026-07-25

### Added
- **Major Framework Completion**: Initial production release of the Vajra Foundation Language Model Framework.
- **Vajra-57M Base Model**: Complete pretraining, evaluation, and packaging of `Vajra-57M` (`90,317,312` parameters).
- **Release Packaging Engine**: `release/package_model.py` generating `model.safetensors`, `pytorch_model.bin`, `README.md` (Model Card), `manifest.json`, and training metrics reports (`training_summary.json/csv/md`).
- **Release Verification System**: `release/verify_package.py` enforcing strict **8/8 rule** verification checks covering metadata, weights, tokenizer, SHA-256 checksums, and structural validity.
- **Deterministic Builds**: Static LF line ending normalization (`newline="\n"`) and Git commit timestamp integration ensuring byte-for-byte reproducible release packages across Windows, Linux, and macOS.
- **Git LFS Integration**: Full Git LFS tracking for `.safetensors` and `.bin` weights.
- **Agent & Memory System**: Multi-agent task orchestration, persistent conversation memory, RAG context building, and knowledge graph storage in `vajra_agent/`.
- **Comprehensive Documentation**: Complete technical guides in `docs/` covering architecture, training, evaluation, tokenizer, dataset pipeline, release pipeline, configuration, contributing, and FAQ.
- **Testing Suite**: **248 unit and integration tests** fully passing across all framework subsystems.

### Fixed
- Fixed `UnicodeEncodeError` on Windows consoles during release verification output.
- Fixed non-deterministic SHA-256 checksum mismatches caused by OS-native CRLF line endings and dynamic wall-clock timestamps.
- Fixed tied weight cloning (`.clone().contiguous().cpu()`) when exporting PyTorch state dicts to Hugging Face `safetensors` format.
