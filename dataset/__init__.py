"""
Vajra Dataset Collection Framework.
Provides robust infrastructure for registering, validating, downloading,
and preparing datasets for pretraining.
"""

from dataset.builder import BinaryDatasetBuilder
from dataset.ingest import DataIngestor
from dataset.statistics import DatasetStatistics
from dataset.sources.registry import DataSourceRegistry, DataSource, create_default_registry
from dataset.sources.synthetic import generate_synthetic_documents, write_synthetic_corpus

__all__ = [
    "BinaryDatasetBuilder",
    "DataIngestor",
    "DatasetStatistics",
    "DataSourceRegistry",
    "DataSource",
    "create_default_registry",
    "generate_synthetic_documents",
    "write_synthetic_corpus",
]
