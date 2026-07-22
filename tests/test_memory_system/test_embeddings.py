"""Tests for MockEmbeddingProvider and embedding interface."""

from vajra_agent.memory.embeddings import MockEmbeddingProvider


def test_mock_embedding_provider():
    provider = MockEmbeddingProvider(dim=32)
    assert provider.dimension == 32

    vec1 = provider.embed_text("Hello world")
    vec2 = provider.embed_text("Hello world")
    vec3 = provider.embed_text("Different query")

    assert len(vec1) == 32
    assert vec1 == vec2  # Deterministic
    assert vec1 != vec3  # Content-dependent
