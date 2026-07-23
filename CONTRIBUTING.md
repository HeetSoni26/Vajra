# Contributing to Vajra-LM & Vajra-Agent

Thank you for your interest in contributing to Vajra-LM and Vajra-Agent!

## Development Setup

1. Clone the repository and navigate to root:
   ```bash
   git clone https://github.com/HeetSoni26/Vajra.git
   cd Vajra
   ```

2. Install in editable mode with development dependencies:
   ```bash
   pip install -e .
   ```

3. Run quality verification tests:
   ```bash
   ruff check .
   pytest
   ```

## Contribution Rules

1. **Architecture Freezing**: Vajra-LM engine components are frozen. Extensions belong inside `vajra_agent/`.
2. **Quality Gate**: All pull requests must pass `ruff check .` with 0 warnings and `pytest` with 100% passing tests.
3. **Type Annotations**: All new functions and methods must include strict Python 3.11 type hints.
4. **Documentation & Examples**: Any new tool or agent module must include unit tests in `tests/` and a runnable example in `examples/`.
