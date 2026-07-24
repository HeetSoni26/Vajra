import json
from pathlib import Path
from typing import Dict, List, Optional
from dataset.mixture.models import DatasetMixture


class MixtureManager:
    """
    Manages loading, saving, and registry of DatasetMixtures.
    """

    def __init__(self, storage_dir: str | Path):
        self.storage_dir = Path(storage_dir)
        self.storage_dir.mkdir(parents=True, exist_ok=True)
        self.mixtures: Dict[str, DatasetMixture] = {}
        self._load_all()

    def _load_all(self):
        for path in self.storage_dir.glob("*.json"):
            try:
                with open(path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    mixture = DatasetMixture.model_validate(data)
                    self.mixtures[mixture.name] = mixture
            except Exception:
                # Log error in real system
                pass

    def save(self, mixture: DatasetMixture) -> None:
        self.mixtures[mixture.name] = mixture
        path = self.storage_dir / f"{mixture.name}.json"
        with open(path, "w", encoding="utf-8") as f:
            f.write(mixture.model_dump_json(indent=2))

    def get(self, name: str) -> Optional[DatasetMixture]:
        return self.mixtures.get(name)

    def list_mixtures(self) -> List[DatasetMixture]:
        return list(self.mixtures.values())

    def export_mixture(self, name: str, export_path: str | Path) -> None:
        mixture = self.get(name)
        if not mixture:
            raise ValueError(f"Mixture {name} not found.")
        with open(export_path, "w", encoding="utf-8") as f:
            f.write(mixture.model_dump_json(indent=2))

    def import_mixture(self, import_path: str | Path) -> DatasetMixture:
        with open(import_path, "r", encoding="utf-8") as f:
            data = json.load(f)
            mixture = DatasetMixture.model_validate(data)
            self.save(mixture)
            return mixture
