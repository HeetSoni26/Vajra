# Changelog

All notable changes to this project will be documented in this file.

## [1.0.0] - 2026-07-22 (Vajra-Agent v1.0.0 Real-World Validation Release)

### Added
- **Real-World Validation Suite (`tests/test_validation/` & `validation/`)**:
  - Multi-language repository scanning & indexing validation across Python FastAPI, Node/Next.js, Go, Rust, and mixed-language projects (`test_real_repo_scanning.py`).
  - 10 Software engineering task validations (`test_engineering_tasks.py`).
  - End-to-end multi-agent scenario pipeline validations (`test_e2e_scenarios.py`).
  - Zero-regression core abstraction test suite (`test_regression_suite.py`).
  - Automated validation runner (`validation/run_validation.py`) producing `validation/validation_report.json`.
- **Validation Documentation (`docs/agent/validation_report.md`)**:
  - Detailed quality metrics report, known strengths, known weaknesses, production deployment notes, and recommendations for future releases.
- **Quality Gate Verification**:
  - Expanded unit test suite to **103/103 passing tests**.
  - **100.0% Validation Task Success Rate** with **0 tool execution failures**.

## [0.9.0] - 2026-07-22 (Vajra-Agent Phase 3 — Memory & Knowledge System)

- MemoryManager, ContextBuilder, Embeddings, Vector Storage, Semantic Retrieval, Knowledge Graph, Retention Policies, Memory Summarizer.

## [0.8.0] - 2026-07-22 (Vajra-Agent Phase 2 — Coding Intelligence)

- Task Planning Engine, Repository Scanner, Workspace Indexer, PythonSandbox, VerificationEngine, ReflectionEngine, ProjectContext, and Coding Workflows.

## [0.7.0] - 2026-07-22 (Vajra-Agent Phase 1 — Core Infrastructure)

- Built initial `vajra_agent/` package: `BaseTool`, `ToolRegistry`, `FunctionParser`, `FoundationAgent` execution loop, `EventBus`, and MCP abstractions.
