"""RetrievalEngine providing semantic search, metadata filtering, and recency weighting."""

from __future__ import annotations

from dataclasses import dataclass
import time
from typing import Any

from vajra_agent.memory.embeddings.base import BaseEmbeddingProvider
from vajra_agent.memory.storage.base import BaseVectorStore, VectorRecord


@dataclass
class RetrievalResult:
    """Structured retrieval search result."""

    record: VectorRecord
    similarity_score: float
    recency_weight: float
    final_score: float


class RetrievalEngine:
    """Semantic retrieval engine combining vector similarity, metadata filtering, and recency weighting."""

    def __init__(
        self,
        embedder: BaseEmbeddingProvider,
        vector_store: BaseVectorStore,
        recency_decay_hours: float = 24.0,
    ) -> None:
        self.embedder = embedder
        self.store = vector_store
        self.recency_decay_hours = recency_decay_hours

    def retrieve(
        self,
        query: str,
        top_k: int = 5,
        metadata_filter: dict[str, Any] | None = None,
        apply_recency: bool = True,
    ) -> list[RetrievalResult]:
        query_vec = self.embedder.embed_text(query)
        raw_matches = self.store.search(query_vec, top_k=top_k * 2, metadata_filter=metadata_filter)

        now = time.time()
        results: list[RetrievalResult] = []

        for rec, sim in raw_matches:
            age_hours = max((now - rec.timestamp) / 3600.0, 0.0)
            recency_weight = 1.0 / (1.0 + (age_hours / self.recency_decay_hours)) if apply_recency else 1.0

            final_score = sim * 0.8 + recency_weight * 0.2 if apply_recency else sim
            results.append(
                RetrievalResult(
                    record=rec,
                    similarity_score=sim,
                    recency_weight=recency_weight,
                    final_score=final_score,
                )
            )

        results.sort(key=lambda item: item.final_score, reverse=True)
        return results[:top_k]
