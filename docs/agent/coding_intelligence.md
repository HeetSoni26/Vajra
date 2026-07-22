# Vajra-Agent Phase 2 — Coding Intelligence Documentation

Phase 2 enhances `Vajra-Agent` with Software Engineering intelligence capabilities: task planning, repository scanning, workspace AST indexing, code execution sandboxing, automatic verification pipelines, self-reflection, project context management, and specialized coding workflows.

## Subsystems Reference

### 1. Task Planning Engine (`vajra_agent/planning/`)
Decomposes user objectives into ordered `PlanStep` dependencies inside a structured `Plan`.

### 2. Repository Scanner (`vajra_agent/repository/`)
Discovers primary language (Python, JS, TS, Go, Rust, Java), framework (FastAPI, Flask, Django, React, Next.js), package manager, entrypoints, and config files.

### 3. Workspace Indexer (`vajra_agent/indexing/`)
Uses Python AST parsing to index classes, functions, methods, signatures, and docstrings across the codebase.

### 4. Code Execution Sandbox (`vajra_agent/sandbox/`)
Executes Python code safely, capturing `stdout`, `stderr`, `exit_code`, generated files, and tracebacks in `SandboxResult`.

### 5. Automatic Verification Engine (`vajra_agent/verification/`)
Runs `ruff`, `pytest`, and `mypy` check suites automatically. Translates lint/test errors into structured agent observation prompts for self-correction.

### 6. Reflection Engine (`vajra_agent/reflection/`)
Generates post-execution critiques (`ReflectionResult`) evaluating reasoning quality, tool efficiency, and verification results.

### 7. Project Context Manager (`vajra_agent/context/`)
Consolidates workspace metadata, active git branch, repository scan context, and recent changes.

### 8. Coding Workflows (`vajra_agent/workflows/`)
High-level reusable engineering workflows: `RepoAnalysisWorkflow`, `BugFixWorkflow`, `FeatureWorkflow`, `RefactorWorkflow`, `TestGenerationWorkflow`.
