"""
Dataset source registry for Vajra Framework.
Provides a centralized catalog of supported data sources with metadata,
licensing, and download configuration.
"""

from dataset.sources.registry import DataSource, DataSourceRegistry

__all__ = ["DataSource", "DataSourceRegistry"]
