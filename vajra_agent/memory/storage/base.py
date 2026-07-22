"""Vector storage models and BaseVectorStore abstract interface."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
import time
import uuid
from typing import Any


@dataclass
class VectorRecord:
    """Represents an individual item in a vector store."""

    text: str
    vector: list[float]
    metadata: dict[str, Any] = field(default_factory=dict)
    id: str = field(default_factory=lambda: f"vec_{uuid.uuid4().hex[:8]}")
    timestamp: float = field(default_factory=time.time)

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "text": self.text,
            "vector": self.vector,
            "metadata": self.metadata,
            "timestamp": self.timestamp,
        }


class BaseVectorStore(ABC):
    """Abstract interface for vector database storage providers."""

    @abstractmethod
    def add(self, record: VectorRecord) -> None:
        """Add a vector record."""
        pass

    @abstractmethod
    def search(
        self,
        query_vector: list[float],
        top_k: int = 5,
        metadata_filter: dict[str, Any] | None = None,
    ) -> list[tuple[VectorRecord, float]]:
        """Search for top_k nearest vector records returning (record, similarity_score)."""
        pass

    @abstractmethod
    def delete(self, record_id: str) -> None:
        """Delete record by ID."""
        pass

    @abstractmethod
    def clear(self) -> None:
        """Clear all stored vector records."""
        pass
