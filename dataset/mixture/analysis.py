from collections import defaultdict
from typing import Any

from dataset.mixture.models import DatasetMixture


class MixtureAnalyzer:
    """
    Analyzes a DatasetMixture to generate human-readable reports and statistics.
    """

    @classmethod
    def generate_report(cls, mixture: DatasetMixture) -> dict[str, Any]:
        """
        Generates an analytical report of the mixture's composition.
        """
        enabled_entries = [e for e in mixture.entries if e.is_enabled]

        language_dist = defaultdict(float)
        domain_dist = defaultdict(float)
        license_dist = defaultdict(float)
        quality_dist = defaultdict(float)

        total_tokens = 0
        total_docs = 0

        for e in enabled_entries:
            language_dist[e.language] += e.weight
            domain_dist[e.domain] += e.weight
            license_dist[e.license] += e.weight
            quality_dist[e.quality_rating.value] += e.weight

            if e.estimated_tokens:
                total_tokens += e.estimated_tokens
            if e.estimated_documents:
                total_docs += e.estimated_documents

        dataset_contribution = {f"{e.name}:{e.version}": e.weight for e in enabled_entries}

        return {
            "mixture_name": mixture.name,
            "version": mixture.version,
            "total_enabled_datasets": len(enabled_entries),
            "estimated_corpus_tokens": total_tokens,
            "estimated_corpus_documents": total_docs,
            "language_distribution": dict(language_dist),
            "domain_distribution": dict(domain_dist),
            "license_distribution": dict(license_dist),
            "quality_distribution": dict(quality_dist),
            "dataset_contribution": dataset_contribution,
        }
