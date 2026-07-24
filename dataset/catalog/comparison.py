from typing import Any

from dataset.metadata.models import DatasetMetadata


class DatasetComparison:
    """
    Utilities for comparing datasets.
    """

    @classmethod
    def compare(cls, datasets: list[DatasetMetadata]) -> dict[str, Any]:
        """
        Compare a list of datasets side-by-side.
        Returns a dictionary representing the comparison matrix.
        """
        if not datasets:
            return {}

        comparison = {
            "name": [ds.name for ds in datasets],
            "version": [ds.version for ds in datasets],
            "license": [ds.license for ds in datasets],
            "language": [ds.language for ds in datasets],
            "domain": [ds.domain for ds in datasets],
            "dataset_type": [ds.dataset_type for ds in datasets],
            "estimated_size_bytes": [ds.estimated_size_bytes for ds in datasets],
            "estimated_tokens": [ds.estimated_tokens for ds in datasets],
            "quality_rating": [ds.quality_rating for ds in datasets],
            "status": [ds.status for ds in datasets],
        }
        return comparison
