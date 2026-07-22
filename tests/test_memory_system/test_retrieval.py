"""Tests for RetrievalEngine semantic search and recency weighting."""

from vajra_agent.memory.embeddings import MockEmbeddingProvider
from vajra_agent.memory.retrieval import RetrievalEngine
from vajra_agent.memory.storage import InMemoryVectorStore, VectorRecord


def test_retrieval_engine_search():
    embedder = MockEmbeddingProvider()
    store = InMemoryVectorStore()
    engine = RetrievalEngine(embedder=embedder, vector_store=store)

    rec1 = VectorRecord(text="FastAPI backend setup", vector=embedder.embed_text("FastAPI backend setup"))
    rec2 = VectorRecord(text="React frontend design", vector=embedder.embed_text("React frontend design"))
    store.add(rec1)
    store.add(rec2)

    results = engine.retrieve("FastAPI backend setup", top_k=1)
    assert len(results) == 1
    assert results[0].record.text == "FastAPI backend setup"
