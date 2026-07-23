# Changelog

All notable changes to this project will be documented in this file.

## [1.0.0] - 2026-07-23 (Vajra v1.0.0 Open Source Production Release)

### Added & Stabilized
- **Phase 1 (Critical Runtime & Architecture Repair)**:
  - Canonicalized model exports on `VajraForCausalLM` and `VajraConfig`.
  - Deprecated redundant model classes with backward compatible aliases.
  - Enhanced Pydantic configuration alias resolution (`rms_norm_eps` / `rmsnorm_eps`, `head_dim`).
  - Fixed training data loader import chains and checkpoint managers.
- **Phase 2 (Security, API Correctness & Production Readiness)**:
  - Hardened `ShellTool` to eliminate unsafe `shell=True` execution.
  - Restricted `PythonTool` and `PythonSandbox` `exec()` environment, blocking hazardous builtins and system imports.
  - Device-agnostic `GradScaler` hardware detection (CUDA, CPU, Apple MPS).
  - Integrated HuggingFace `AutoConfig` and `AutoModelForCausalLM` registration with `strict=True` weight loading validation.
  - Hardened Docker containers with non-root `vajra` user (UID/GID 10001).
- **Phase 3 (Medium Severity & Production Completeness)**:
  - Added missing `__init__.py` files across all python packages.
  - Migrated Pydantic V2 configurations to `ConfigDict` and `SettingsConfigDict`.
  - Implemented production middleware (`CORS`, `X-Request-ID`, execution profiling, error logging).
  - Implemented working `LocalBackend`, `HTTPBackend`, `HuggingFaceBackend`, and `GitBackend` dataset downloaders.
  - Converted API endpoints to async with SSE streaming generators.
  - Clarified label shifting semantics in `VajraForCausalLM.forward`.
- **Phase 4 (Final Open Source Release Preparation)**:
  - Updated `SECURITY.md`, `CITATION.cff`, `pyproject.toml` extras, and pytest infrastructure `conftest.py`.
  - Reached **100% test pass rate (180/180 passed)** and **0 Ruff lint errors**.
- **Phase 5 (Dataset Engineering & Training Preparation)**:
  - Added `DataSourceRegistry` with 9 curated open pretraining sources (FineWeb-Edu, The Stack v2, OpenWebMath, peS2o, Wikipedia, Gutenberg, etc.), YAML serialisation, domain/tag/license filtering, and download manifest generation.
  - Added `SyntheticDataGenerator` for 7-domain synthetic corpus generation (CI/dev, reproducible, no downloads required).
  - Added `DatasetStatistics` engine: token distribution, vocab coverage, top-K frequency analysis, and integrity validation.
  - Added training preset configs for **Vajra-125M** and **Vajra-370M** (model YAML + training YAML, Chinchilla-scaled token targets).
  - Added `configs/data/sources.yaml` and enhanced `configs/data/dataset_mix.yaml` with per-tier weight overrides and sampling strategy.
  - Added `scripts/prepare_dataset.py`: end-to-end orchestrator (synthetic + production modes).
  - Added `scripts/verify_training_readiness.py`: 5-stage pre-flight verification (model, dataset, forward/backward, optimizer, checkpoint round-trip).
  - Expanded unit test suite to **213/213 passing tests** (33 new Phase 5 tests added).

## [0.9.0] - 2026-07-22 (Vajra-Agent Phase 3 — Memory & Knowledge System)

- MemoryManager, ContextBuilder, Embeddings, Vector Storage, Semantic Retrieval, Knowledge Graph, Retention Policies, Memory Summarizer.

## [0.8.0] - 2026-07-22 (Vajra-Agent Phase 2 — Coding Intelligence)

- Task Planning Engine, Repository Scanner, Workspace Indexer, PythonSandbox, VerificationEngine, ReflectionEngine, ProjectContext, and Coding Workflows.

## [0.7.0] - 2026-07-22 (Vajra-Agent Phase 1 — Core Infrastructure)

- Built initial `vajra_agent/` package: `BaseTool`, `ToolRegistry`, `FunctionParser`, `FoundationAgent` execution loop, `EventBus`, and MCP abstractions.
