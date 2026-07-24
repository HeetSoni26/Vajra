import tempfile

import pytest

from dataset.catalog.comparison import DatasetComparison
from dataset.catalog.reports import CatalogReportGenerator
from dataset.catalog.scoring import QualityScoringFramework, ScoringCriterion
from dataset.catalog.search import DatasetSearch
from dataset.metadata.licenses import LicenseCategory, LicenseValidator
from dataset.metadata.models import DatasetMetadata, DatasetType, QualityRating
from dataset.registry.registry import DatasetRegistry


@pytest.fixture
def temp_manifests_dir():
    with tempfile.TemporaryDirectory() as d:
        yield d


@pytest.fixture
def sample_datasets():
    ds1 = DatasetMetadata(
        name="dataset1",
        version="1.0.0",
        description="desc",
        source="local",
        download_method="local",
        license="Apache 2.0",
        domain="code",
        format="jsonl",
        dataset_type=DatasetType.PRETRAINING,
        quality_rating=QualityRating.HIGH,
        estimated_size_bytes=100,
    )
    ds2 = DatasetMetadata(
        name="dataset2",
        version="1.0.0",
        description="desc",
        source="local",
        download_method="local",
        license="CC-BY-NC",
        domain="math",
        format="parquet",
        dataset_type=DatasetType.INSTRUCTION,
        quality_rating=QualityRating.MEDIUM,
        tags=["reasoning"],
    )
    return [ds1, ds2]


def test_license_validator():
    assert LicenseValidator.classify("Apache 2.0") == LicenseCategory.COMMERCIALLY_USABLE
    assert LicenseValidator.classify("CC-BY-NC") == LicenseCategory.RESEARCH_ONLY
    assert LicenseValidator.classify("Unknown License") == LicenseCategory.UNKNOWN


def test_search(temp_manifests_dir, sample_datasets):
    registry = DatasetRegistry(manifests_dir=temp_manifests_dir)
    for ds in sample_datasets:
        registry.register(ds)

    searcher = DatasetSearch(registry)

    # Search by domain
    results = searcher.search(domain="math")
    assert len(results) == 1
    assert results[0].name == "dataset2"

    # Search by type
    results = searcher.search(dataset_type="pretraining")
    assert len(results) == 1
    assert results[0].name == "dataset1"

    # Search by tag
    results = searcher.search(tags=["reasoning"])
    assert len(results) == 1
    assert results[0].name == "dataset2"


def test_comparison(sample_datasets):
    comp = DatasetComparison.compare(sample_datasets)
    assert len(comp["name"]) == 2
    assert "dataset1" in comp["name"]
    assert comp["license"] == ["Apache 2.0", "CC-BY-NC"]


def test_report_generation(temp_manifests_dir, sample_datasets):
    registry = DatasetRegistry(manifests_dir=temp_manifests_dir)
    for ds in sample_datasets:
        registry.register(ds)

    generator = CatalogReportGenerator(registry)
    report = generator.generate_summary_report()

    assert report["total_datasets"] == 2
    assert report["license_distribution"]["Apache 2.0"] == 1
    assert report["missing_metadata"]["no_homepage"] == 2


class MockScoringCriterion(ScoringCriterion):
    def evaluate(self, metadata):
        return 0.9 if metadata.domain == "code" else 0.4


def test_scoring_framework(sample_datasets):
    framework = QualityScoringFramework()
    framework.add_criterion(MockScoringCriterion("DomainScore", 1.0))

    ds1_scores = framework.score_dataset(sample_datasets[0])
    assert ds1_scores["DomainScore"] == 0.9
    assert ds1_scores["total_score"] == 0.9
    assert (
        QualityScoringFramework.map_score_to_rating(ds1_scores["total_score"]) == QualityRating.HIGH
    )
