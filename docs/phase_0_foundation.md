# Phase 0 — Project Foundation

## Deliverables included

- Reproducible environment files: `environment.yml`, `Dockerfile`, `pyproject.toml`
- CI skeleton: `.github/workflows/ci.yml`
- Project commands: `Makefile`
- Development hygiene: `.gitignore`, tests, lint configuration

## Validation checklist

- `pytest` passes
- `ruff check .` passes
- CUDA instance can import Torch and report GPU availability
- FlashAttention install is validated separately on the target GPU host
- W&B credentials are supplied through environment variables only
