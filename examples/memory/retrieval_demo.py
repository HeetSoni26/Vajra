"""Retrieval engine demonstration combining similarity and recency."""

from vajra_agent import InMemoryVectorStore, MockEmbeddingProvider, RetrievalEngine, VectorRecord


def main():
    embedder = MockEmbeddingProvider()
    store = InMemoryVectorStore()
    engine = RetrievalEngine(embedder=embedder, vector_store=store, recency_decay_hours=1.0)

    rec1 = VectorRecord(text="Old auth design", vector=embedder.embed_text("auth design"), timestamp=0)
    rec2 = VectorRecord(text="New JWT auth design", vector=embedder.embed_text("auth design"))
    store.add(rec1)
    store.add(rec2)

    results = engine.retrieve("auth design", top_k=2, apply_recency=True)

    print("Retrieval Results (with Recency Weighting):")
    for res in results:
        print(f"  Score: {res.final_score:.4f} | Recency: {res.recency_weight:.4f} | Text: {res.record.text}")


if __name__ == "__main__":
    main()
