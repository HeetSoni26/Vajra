from typing import List, Optional
from dataset.metadata.models import DatasetMetadata
from dataset.registry.registry import DatasetRegistry

class DatasetSearch:
    """
    Provides robust search capabilities across the DatasetRegistry.
    """
    def __init__(self, registry: DatasetRegistry):
        self.registry = registry
        
    def search(
        self, 
        query: Optional[str] = None,
        language: Optional[str] = None,
        domain: Optional[str] = None,
        license: Optional[str] = None,
        tags: Optional[List[str]] = None,
        dataset_type: Optional[str] = None,
        quality_rating: Optional[str] = None
    ) -> List[DatasetMetadata]:
        """
        Search and filter registered datasets based on multiple criteria.
        """
        all_datasets = self.registry.list_datasets()
        results = []
        
        for ds in all_datasets:
            if query and query.lower() not in ds.name.lower() and query.lower() not in ds.description.lower():
                continue
            if language and ds.language.lower() != language.lower():
                continue
            if domain and ds.domain.lower() != domain.lower():
                continue
            if license and ds.license.lower() != license.lower():
                continue
            if dataset_type and ds.dataset_type != dataset_type:
                continue
            if quality_rating and ds.quality_rating != quality_rating:
                continue
            if tags:
                if not set(tags).issubset(set(ds.tags)):
                    continue
            results.append(ds)
            
        return results
