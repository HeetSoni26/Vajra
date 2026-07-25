# Vajra Official Release Checklist

This document details the step-by-step procedure required for tagging and releasing a production version of the **Vajra** framework.

---

## Phase 1: Pre-Release Audit & Version Bumping

- [ ] **Version Bump**: Update version string (`v1.0.0`) in `pyproject.toml`, `setup.py`, `configs/`, and release scripts.
- [ ] **Git Working Tree Check**: Assert `git status` shows zero uncommitted changes.
- [ ] **Documentation Audit**: Verify links across `README.md`, `VISION.md`, `ROADMAP.md`, `CHANGELOG.md`, and `docs/`.
- [ ] **Training Report Verification**: Confirm `docs/models/<model>-training-report.md` matches pretraining metrics.

---

## Phase 2: Build & Verification Pipeline

- [ ] **Package Generation**: Execute release model packaging:
  ```bash
  python -m release.package_model \
      --checkpoint checkpoints/pretrain_tiny/best.pt \
      --config configs/training/pretrain_tiny.yaml \
      --output-dir release/vajra-57m \
      --model-name vajra-57m
  ```
- [ ] **Deterministic Line Ending Normalization**: Ensure all text, JSON, CSV, and Markdown files have `\n` line endings.
- [ ] **Package Integrity Verification (8/8 Rule)**: Execute verifier:
  ```bash
  python -m release.verify_package --package-dir release/vajra-57m
  ```
  *Must report `Package Verification Status: [SUCCESS]` (Passed 8/8 checks).*

- [ ] **PyTest Validation**: Run full automated test suite:
  ```bash
  pytest
  ```
  *Must report 100% passing status (248 passed).*

---

## Phase 3: Git Tagging & Release Publication

- [ ] **Stage & Commit Release Artifacts**:
  ```bash
  git add .
  git commit -m "docs: prepare repository for production GitHub v1.0.0 release"
  git push origin main
  ```
- [ ] **Create Annotated Git Tag**:
  ```bash
  git tag -a v1.0.0 -m "Vajra Version 1.0.0 Production Release"
  git push origin v1.0.0
  ```
- [ ] **GitHub Release Page Publication**:
  - Copy release body content from `docs/releases/v1.0.0.md`.
  - Target tag: `v1.0.0`.
  - Attach release asset manifest references.

---

## Phase 4: Post-Release Validation & Announcements

- [ ] **Clean Clone Verification**: Perform a fresh `git clone` into a temporary directory and execute `python -m release.verify_package --package-dir release/vajra-57m`.
- [ ] **Hugging Face Model Hub Sync**: (Optional) Publish model weights to Hugging Face repository.
- [ ] **Community Announcement**: Publish release announcements to GitHub Discussions and community forums.
