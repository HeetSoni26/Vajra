"""
Dataset Source Registry — centralised catalog of data sources for pretraining.

Each DataSource entry stores metadata (name, URL pattern, license, domain,
estimated size, format) and validation helpers. The registry supports:
  • Registration / deregistration of sources
  • Filtering by domain, license, or tag
  • Serialisation to / from YAML for reproducible experiments
  • Generating download manifests
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from utils.logging import setup_logger

logger = setup_logger("data_source_registry")


@dataclass
class DataSource:
    """Metadata for a single pretraining data source."""

    name: str
    url: str
    domain: str  # e.g., "web", "code", "math", "science", "books", "wikipedia"
    license: str  # e.g., "CC-BY-SA-4.0", "Apache-2.0", "Public Domain"
    format: str = "jsonl"  # jsonl | txt | parquet | csv | arrow
    estimated_size_gb: float = 0.0
    language: str = "en"
    description: str = ""
    tags: list[str] = field(default_factory=list)
    enabled: bool = True
    priority: int = 1  # Lower = higher priority during sampling
    quality_tier: str = "standard"  # standard | high | experimental
    download_method: str = "http"  # http | hf | git | gcs | s3

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "url": self.url,
            "domain": self.domain,
            "license": self.license,
            "format": self.format,
            "estimated_size_gb": self.estimated_size_gb,
            "language": self.language,
            "description": self.description,
            "tags": self.tags,
            "enabled": self.enabled,
            "priority": self.priority,
            "quality_tier": self.quality_tier,
            "download_method": self.download_method,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> DataSource:
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})


class DataSourceRegistry:
    """Centralised registry of pretraining data sources."""

    def __init__(self) -> None:
        self._sources: dict[str, DataSource] = {}

    # ------------------------------------------------------------------
    # Registration
    # ------------------------------------------------------------------
    def register(self, source: DataSource) -> None:
        """Register a new data source (or overwrite existing)."""
        self._sources[source.name] = source
        logger.debug(f"Registered data source: {source.name}")

    def deregister(self, name: str) -> None:
        """Remove a data source by name."""
        self._sources.pop(name, None)

    # ------------------------------------------------------------------
    # Querying
    # ------------------------------------------------------------------
    def get(self, name: str) -> DataSource | None:
        return self._sources.get(name)

    def list_all(self) -> list[DataSource]:
        return list(self._sources.values())

    def list_enabled(self) -> list[DataSource]:
        return [s for s in self._sources.values() if s.enabled]

    def filter_by_domain(self, domain: str) -> list[DataSource]:
        return [s for s in self._sources.values() if s.domain == domain]

    def filter_by_license(self, license_id: str) -> list[DataSource]:
        return [s for s in self._sources.values() if s.license == license_id]

    def filter_by_tag(self, tag: str) -> list[DataSource]:
        return [s for s in self._sources.values() if tag in s.tags]

    def filter_by_quality_tier(self, tier: str) -> list[DataSource]:
        return [s for s in self._sources.values() if s.quality_tier == tier]

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------
    def summary(self) -> dict[str, Any]:
        """Generate a summary of the registry."""
        sources = self.list_all()
        enabled = [s for s in sources if s.enabled]
        domains = {}
        for s in enabled:
            domains.setdefault(s.domain, []).append(s.name)

        return {
            "total_sources": len(sources),
            "enabled_sources": len(enabled),
            "total_estimated_gb": round(sum(s.estimated_size_gb for s in enabled), 2),
            "domains": {k: len(v) for k, v in domains.items()},
            "domain_details": domains,
        }

    # ------------------------------------------------------------------
    # Serialisation
    # ------------------------------------------------------------------
    def save_yaml(self, path: str | Path) -> None:
        """Serialise registry to YAML."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        data = {"sources": [s.to_dict() for s in self._sources.values()]}
        with path.open("w", encoding="utf-8") as f:
            yaml.safe_dump(data, f, default_flow_style=False, sort_keys=False)
        logger.info(f"Registry saved to {path}")

    def load_yaml(self, path: str | Path) -> None:
        """Load sources from YAML, merging into current registry."""
        path = Path(path)
        if not path.exists():
            logger.warning(f"Registry file not found: {path}")
            return
        with path.open("r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
        for entry in data.get("sources", []):
            self.register(DataSource.from_dict(entry))
        logger.info(f"Loaded {len(data.get('sources', []))} sources from {path}")

    # ------------------------------------------------------------------
    # Download manifest generation
    # ------------------------------------------------------------------
    def generate_download_manifest(self) -> list[dict[str, Any]]:
        """Generate ordered download manifest from enabled sources."""
        enabled = sorted(self.list_enabled(), key=lambda s: s.priority)
        return [
            {
                "name": s.name,
                "url": s.url,
                "domain": s.domain,
                "format": s.format,
                "download_method": s.download_method,
                "estimated_size_gb": s.estimated_size_gb,
            }
            for s in enabled
        ]


def create_default_registry() -> DataSourceRegistry:
    """Create a registry pre-populated with recommended open pretraining sources."""
    registry = DataSourceRegistry()

    # ── Web ──
    registry.register(DataSource(
        name="fineweb-edu",
        url="https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu",
        domain="web",
        license="ODC-By-1.0",
        format="parquet",
        estimated_size_gb=500.0,
        description="High-quality educational web content filtered from CommonCrawl.",
        tags=["pretraining", "web", "educational"],
        quality_tier="high",
        download_method="hf",
    ))
    registry.register(DataSource(
        name="cc-refined",
        url="https://huggingface.co/datasets/tiiuae/falcon-refinedweb",
        domain="web",
        license="ODC-By-1.0",
        format="parquet",
        estimated_size_gb=600.0,
        description="Refined web corpus with deduplication and quality filtering.",
        tags=["pretraining", "web"],
        download_method="hf",
    ))

    # ── Code ──
    registry.register(DataSource(
        name="the-stack-v2-dedup",
        url="https://huggingface.co/datasets/bigcode/the-stack-v2-dedup",
        domain="code",
        license="Various",
        format="parquet",
        estimated_size_gb=300.0,
        description="Deduplicated source code across 600+ programming languages.",
        tags=["pretraining", "code"],
        quality_tier="high",
        download_method="hf",
    ))

    # ── Math ──
    registry.register(DataSource(
        name="open-web-math",
        url="https://huggingface.co/datasets/open-web-math/open-web-math",
        domain="math",
        license="CC-BY-SA-4.0",
        format="parquet",
        estimated_size_gb=14.0,
        description="Mathematical web content curated for LLM pretraining.",
        tags=["pretraining", "math"],
        quality_tier="high",
        download_method="hf",
    ))

    # ── Science ──
    registry.register(DataSource(
        name="arxiv-abstracts",
        url="https://huggingface.co/datasets/ccdv/arxiv-abstract",
        domain="science",
        license="CC0-1.0",
        format="parquet",
        estimated_size_gb=3.0,
        description="ArXiv paper abstracts across all scientific domains.",
        tags=["pretraining", "science", "academic"],
        download_method="hf",
    ))
    registry.register(DataSource(
        name="peS2o",
        url="https://huggingface.co/datasets/allenai/peS2o",
        domain="science",
        license="ODC-By-1.0",
        format="parquet",
        estimated_size_gb=50.0,
        description="Scientific paper corpus from Semantic Scholar.",
        tags=["pretraining", "science"],
        quality_tier="high",
        download_method="hf",
    ))

    # ── Books ──
    registry.register(DataSource(
        name="gutenberg",
        url="https://huggingface.co/datasets/manu/project_gutenberg",
        domain="books",
        license="Public Domain",
        format="parquet",
        estimated_size_gb=8.0,
        description="Public domain books from Project Gutenberg.",
        tags=["pretraining", "books", "literature"],
        download_method="hf",
    ))

    # ── Wikipedia ──
    registry.register(DataSource(
        name="wikipedia-en",
        url="https://huggingface.co/datasets/wikimedia/wikipedia",
        domain="wikipedia",
        license="CC-BY-SA-4.0",
        format="parquet",
        estimated_size_gb=21.0,
        description="English Wikipedia articles (latest dump).",
        tags=["pretraining", "wikipedia", "knowledge"],
        quality_tier="high",
        download_method="hf",
    ))

    # ── Technical ──
    registry.register(DataSource(
        name="stackexchange",
        url="https://huggingface.co/datasets/HuggingFaceTB/stack-exchange-preferences",
        domain="technical",
        license="CC-BY-SA-4.0",
        format="parquet",
        estimated_size_gb=10.0,
        description="Stack Exchange Q&A pairs across technical domains.",
        tags=["pretraining", "technical", "qa"],
        download_method="hf",
    ))

    return registry
