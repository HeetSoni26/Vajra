"""InMemoryVectorStore with Cosine Similarity search."""

from __future__ import annotations

from typing import Any
from vajra_agent.memory.storage.base import BaseVectorStore, VectorRecord


def _cosine_similarity(a: list[float], b: list[float]) -> float:
    if len(a) != len(b):
        return 0.0
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = sum(x * x for x in a) ** 0.5
    norm_b = sum(y * y for y in b) ** 0.5
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)


class InMemoryVectorStore(BaseVectorStore):
    """In-memory vector store supporting cosine similarity and metadata filtering."""

    def __init__(self) -> None:
        self._records: list[VectorRecord] = []

    def add(self, record: VectorRecord) -> None:
        self._records.append(record)

    def search(
        self,
        query_vector: list[float],
        top_k: int = 5,
        metadata_filter: dict[str, Any] | None = None,
    ) -> list[tuple[VectorRecord, float]]:
        scored = []
        for rec in self._records:
            # Metadata filter check
            if metadata_filter:
                match = all(rec.metadata.get(k) == v for k, v in metadata_filter.items())
                if not match:
                    continue

            sim = _cosine_similarity(query_vector, rec.vector)
            scored.append((rec, sim))

        scored.sort(key=lambda item: item[1], reverse=True)
        return scored[:top_k]

    def delete(self, record_id: str) -> None:
        self._records = [r for r in self._records if r.id != record_id]

    def clear(self) -> None:
        self._records.clear()

    def __len__(self) -> int:
        return len(self._records)
