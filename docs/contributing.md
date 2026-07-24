# Contributing to Vajra

[Overview](../README.md) | [Architecture](architecture.md) | [Roadmap](roadmap.md)

---

## Overview

Thank you for your interest in contributing to **Vajra**! We welcome community contributions in architecture optimization, dataset loader scaling, bug fixes, documentation, and evaluation benchmark tooling.

---

## Development Workflow

### 1. Repository Setup
```bash
git clone https://github.com/HeetSoni26/Vajra.git
cd Vajra
git lfs install
python -m venv venv
source venv/bin/activate  # Or .\venv\Scripts\Activate.ps1 on Windows
pip install -r requirements.txt
pip install -e .
```

### 2. Code Standards & Formatting
- **PEP 8**: Adhere to Python standard style guidelines.
- **Typing**: Use static type hints (`mypy` / Python 3.10+ annotations).
- **Docstrings**: Google-style docstrings for public functions and classes.
- **Ruff / Linting**: Run `ruff check .` prior to opening a PR.

### 3. Running Test Suites
Before submitting a pull request, ensure the full PyTest suite and release verifier pass:

```bash
# Run pytest (248 tests)
pytest

# Verify package integrity
python -m release.verify_package --package-dir release/vajra-57m
```

---

## Pull Request Guidelines

1. **Feature Branch**: Create a feature branch off `main` (e.g. `feat/rope-scaling` or `fix/tokenizer-pad`).
2. **Atomic Commits**: Keep commit messages concise, descriptive, and imperative (`feat: add DPO loss implementation`).
3. **Tests**: Include unit tests under `tests/` for any new functionality or bug fixes.
4. **Documentation**: Update corresponding `docs/` files if introducing new configuration options or scripts.
