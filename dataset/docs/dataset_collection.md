# Vajra Dataset Collection Framework

The Vajra Dataset Collection Framework is a production-grade infrastructure designed for discovering, registering, downloading, and validating datasets for Vajra-LM model training.

## Architecture

The framework consists of the following components:
- **Registry**: Manages dataset identity and structural expectations (manifests).
- **Download Manager**: Resilient, resumable downloader supporting multiple backends.
- **Cache Manager**: Tracks partial downloads and prevents re-downloading existing chunks.
- **Validators**: Verifies dataset integrity via checksums and manifest expectations.
- **Configuration**: Globally configurable via environment variables (`VAJRA_DATASET_...`).

## Directory Structure

```text
dataset/
├── registry/       # Dataset registration and discovery
├── downloaders/    # DownloadManager and backends (HTTP, HF, Local)
├── metadata/       # Pydantic schemas (DatasetMetadata)
├── manifests/      # JSON manifests for registered datasets
├── cache/          # Download state tracking
├── configs/        # Global framework settings
├── validators/     # Integrity and compliance checking
├── utils/          # Hashing and structured logging
├── scripts/        # CLI tools (manage_dataset.py)
└── docs/           # Documentation
```

## Adding a New Dataset

1. Construct a `DatasetMetadata` object.
2. Use the `DatasetRegistry.register()` method to store its manifest.
3. Commit the generated JSON manifest in `dataset/manifests/`.

## Validation

Datasets can be validated using the `DatasetValidator` to ensure all expected files are present and match their SHA256 checksums.

```bash
python dataset/scripts/manage_dataset.py validate <name> <version>
```
