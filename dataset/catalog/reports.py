from collections import Counter
from typing import Any

from dataset.metadata.licenses import LicenseValidator
from dataset.registry.registry import DatasetRegistry


class CatalogReportGenerator:
    """
    Generates summary reports for the dataset catalog.
    """

    def __init__(self, registry: DatasetRegistry):
        self.registry = registry

    def generate_summary_report(self) -> dict[str, Any]:
        """
        Generates a statistical summary of the registered datasets.
        """
        datasets = self.registry.list_datasets()

        report = {
            "total_datasets": len(datasets),
            "license_distribution": Counter(),
            "license_category_distribution": Counter(),
            "language_distribution": Counter(),
            "domain_distribution": Counter(),
            "quality_distribution": Counter(),
            "missing_metadata": {"no_homepage": 0, "no_citation": 0, "no_size": 0, "no_tokens": 0},
        }

        for ds in datasets:
            report["license_distribution"][ds.license] += 1

            category = LicenseValidator.classify(ds.license)
            report["license_category_distribution"][category.value] += 1

            report["language_distribution"][ds.language] += 1
            report["domain_distribution"][ds.domain] += 1
            report["quality_distribution"][ds.quality_rating] += 1

            if not ds.homepage:
                report["missing_metadata"]["no_homepage"] += 1
            if not ds.citation:
                report["missing_metadata"]["no_citation"] += 1
            if ds.estimated_size_bytes == 0:
                report["missing_metadata"]["no_size"] += 1
            if ds.estimated_tokens is None:
                report["missing_metadata"]["no_tokens"] += 1

        # Convert Counters to dicts for clean output
        for key in [
            "license_distribution",
            "license_category_distribution",
            "language_distribution",
            "domain_distribution",
            "quality_distribution",
        ]:
            report[key] = dict(report[key])

        return report
