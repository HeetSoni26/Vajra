import json
from pathlib import Path
from typing import List, Optional

from dataset.configs.settings import config
from dataset.metadata.models import DatasetMetadata
from dataset.utils.exceptions import DatasetRegistrationError
from dataset.utils.logging import logger

class DatasetRegistry:
    """
    Manages the registration and discovery of datasets.
    Backed by local JSON manifests in the configured manifests_dir.
    """

    def __init__(self, manifests_dir: Optional[str] = None):
        self.manifests_dir = Path(manifests_dir or config.manifests_dir)
        self.manifests_dir.mkdir(parents=True, exist_ok=True)
        self._cache = {}
        self._load_all()

    def _get_manifest_path(self, name: str, version: str) -> Path:
        """Returns the expected path for a dataset manifest."""
        # Sanitize filename
        safe_name = name.replace("/", "_").replace("\\", "_")
        return self.manifests_dir / f"{safe_name}_v{version}.json"

    def _load_all(self):
        """Loads all JSON manifests from the directory into memory."""
        self._cache.clear()
        if not self.manifests_dir.exists():
            return
            
        for filepath in self.manifests_dir.glob("*.json"):
            try:
                with open(filepath, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    metadata = DatasetMetadata(**data)
                    key = f"{metadata.name}:{metadata.version}"
                    self._cache[key] = metadata
            except Exception as e:
                logger.warning(f"Failed to load manifest {filepath}: {e}")

    def register(self, metadata: DatasetMetadata, overwrite: bool = False) -> None:
        """
        Registers a new dataset. Persists to disk.
        """
        key = f"{metadata.name}:{metadata.version}"
        if key in self._cache and not overwrite:
            raise DatasetRegistrationError(f"Dataset {key} is already registered.")

        manifest_path = self._get_manifest_path(metadata.name, metadata.version)
        try:
            with open(manifest_path, "w", encoding="utf-8") as f:
                f.write(metadata.model_dump_json(indent=2))
            self._cache[key] = metadata
            logger.info(f"Successfully registered dataset: {key}")
        except Exception as e:
            logger.error(f"Failed to write manifest for {key}: {e}")
            raise DatasetRegistrationError(f"Failed to write manifest: {e}") from e

    def get(self, name: str, version: str) -> DatasetMetadata:
        """
        Retrieves a dataset's metadata by name and version.
        """
        key = f"{name}:{version}"
        if key not in self._cache:
            raise DatasetRegistrationError(f"Dataset not found: {key}")
        return self._cache[key]

    def list_datasets(self, tags: Optional[List[str]] = None) -> List[DatasetMetadata]:
        """
        Lists all registered datasets, optionally filtering by tags.
        """
        datasets = list(self._cache.values())
        if tags:
            tag_set = set(tags)
            datasets = [ds for ds in datasets if tag_set.issubset(set(ds.tags))]
        return sorted(datasets, key=lambda x: x.name)
