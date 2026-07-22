# Phase 2 — Dataset Pipeline

## Included files

- `configs/data/dataset_mix.yaml`
- `configs/data/preprocessing.yaml`
- `dataset/run_pipeline.py`
- `dataset/processing/*`
- `dataset/tokenize_dataset.py`
- `dataset/verify_dataset.py`
- `dataset/manifests/v1.0_manifest.json`

## Pipeline stages

Extraction → normalization → language filtering → quality filtering → toxicity filtering → deduplication → scoring → contamination check → tokenization → manifest.

The scaffold uses placeholders for stages that require large datasets or external tools. Fill each stage with streaming implementations before running at scale.
