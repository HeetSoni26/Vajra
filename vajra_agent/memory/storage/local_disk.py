"""LocalDiskVectorStore persisting vector records to disk JSON."""

from __future__ import annotations

import json
from pathlib import Path

from vajra_agent.memory.storage.base import VectorRecord
from vajra_agent.memory.storage.in_memory import InMemoryVectorStore


class LocalDiskVectorStore(InMemoryVectorStore):
    """Disk-backed vector store persisting state across sessions in a JSON file."""

    def __init__(self, storage_file: str | Path) -> None:
        super().__init__()
        self.storage_file = Path(storage_file)
        self.load()

    def save(self) -> None:
        self.storage_file.parent.mkdir(parents=True, exist_ok=True)
        data = [r.to_dict() for r in self._records]
        self.storage_file.write_text(json.dumps(data, indent=2), encoding="utf-8")

    def load(self) -> None:
        if not self.storage_file.exists():
            return
        try:
            raw = self.storage_file.read_text(encoding="utf-8")
            data = json.loads(raw)
            self._records = [
                VectorRecord(
                    id=item["id"],
                    text=item["text"],
                    vector=item["vector"],
                    metadata=item.get("metadata", {}),
                    timestamp=item.get("timestamp", 0.0),
                )
                for item in data
            ]
        except Exception:
            pass

    def add(self, record: VectorRecord) -> None:
        super().add(record)
        self.save()

    def delete(self, record_id: str) -> None:
        super().delete(record_id)
        self.save()

    def clear(self) -> None:
        super().clear()
        self.save()
