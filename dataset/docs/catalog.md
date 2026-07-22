# Vajra Dataset Catalog & Validation Framework

This document outlines the usage of the Dataset Catalog and Validation Framework.

## Catalog Architecture
The Dataset Catalog extends the Dataset Registry with detailed metadata and utilities designed for exploratory analysis, quality scoring, and dataset comparison prior to ingestion.

Components:
- **Models (`dataset/metadata/models.py`)**: Extends base `DatasetMetadata` with fields like `dataset_type`, `maintenance_status`, `quality_rating`, `estimated_tokens`, and `num_documents`.
- **License Classification (`dataset/metadata/licenses.py`)**: Automatically categorizes open-source licenses into commercial/research-only tiers via `LicenseValidator`.
- **Quality Scoring (`dataset/catalog/scoring.py`)**: A modular evaluation engine `QualityScoringFramework` using `ScoringCriterion` plugins to rank dataset health.
- **Search (`dataset/catalog/search.py`)**: The `DatasetSearch` engine enables multi-dimensional filtering across domain, language, license, and quality.
- **Comparison (`dataset/catalog/comparison.py`)**: The `DatasetComparison` utility renders feature matrices of selected datasets.
- **Reports (`dataset/catalog/reports.py`)**: The `CatalogReportGenerator` builds analytical summaries of the entire catalog state.

## License Classification
Licenses are statically mapped to `LicenseCategory` enums:
- `COMMERCIALLY_USABLE`: e.g. Apache 2.0, MIT, BSD.
- `RESEARCH_ONLY`: e.g. CC-BY-NC.
- `RESTRICTED`: e.g. specific proprietary variants.
- `UNKNOWN`: Custom or undefined licenses.

## Searching the Catalog
You can search datasets by various criteria using the CLI:
```bash
python dataset/scripts/manage_dataset.py search --domain math --dataset-type pretraining --language en
```

## Comparing Datasets
Compare side-by-side using the CLI:
```bash
python dataset/scripts/manage_dataset.py compare the_stack:v1.0 fineweb:v2.0
```

## Reporting
Generate a summary report of all registered datasets:
```bash
python dataset/scripts/manage_dataset.py report
```
