"""
Vajra Dataset Collection Framework.
Provides robust infrastructure for registering, validating, downloading,
and preparing datasets for pretraining.
"""

from dataset.builder import BinaryDatasetBuilder
from dataset.ingest import DataIngestor
from dataset.sources.registry import DataSource, DataSourceRegistry, create_default_registry
from dataset.sources.synthetic import generate_synthetic_documents, write_synthetic_corpus
from dataset.statistics import DatasetStatistics

__all__ = [
    "BinaryDatasetBuilder",
    "DataIngestor",
    "DataSource",
    "DataSourceRegistry",
    "DatasetStatistics",
    "create_default_registry",
    "generate_synthetic_documents",
    "write_synthetic_corpus",
]
