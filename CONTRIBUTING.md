# Contributing to Vajra

Thank you for contributing to Vajra! We welcome pull requests, bug reports, feature requests, and documentation enhancements.

---

## Code of Conduct

All contributors are expected to adhere to our [Code of Conduct](CODE_OF_CONDUCT.md).

---

## How to Contribute

### 1. Reporting Bugs
- Search existing GitHub Issues before submitting a new report.
- Use the [Bug Report Template](.github/ISSUE_TEMPLATE/bug_report.md).
- Include minimal reproducible code examples, stack trace, and environment details.

### 2. Suggesting Features
- Open a feature request using the [Feature Request Template](.github/ISSUE_TEMPLATE/feature_request.md).
- Clearly explain the problem the feature solves and proposed API designs.

### 3. Submitting Pull Requests
1. Fork the repo and create your feature branch: `git checkout -b feat/my-feature`.
2. Follow PEP 8 guidelines and static type annotations.
3. Verify that all tests pass (`pytest`) and release verification succeeds (`python -m release.verify_package --package-dir release/vajra-57m`).
4. Submit a Pull Request targeting `main` with a clear summary of changes.

---

## Testing Requirements
- Unit tests under `tests/` must cover any new architectural components or utility scripts.
- Current validation threshold: **248 passed** tests.
