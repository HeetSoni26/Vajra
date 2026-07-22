"""Abstract BaseEmbeddingProvider interface for text embedding models."""

from __future__ import annotations

from abc import ABC, abstractmethod


class BaseEmbeddingProvider(ABC):
    """Abstract interface for text embedding providers."""

    @property
    @abstractmethod
    def dimension(self) -> int:
        """Vector embedding output dimension."""
        pass

    @abstractmethod
    def embed_text(self, text: str) -> list[float]:
        """Embed a single text string into a vector float list."""
        pass

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        """Embed a batch of text strings."""
        return [self.embed_text(t) for t in texts]
