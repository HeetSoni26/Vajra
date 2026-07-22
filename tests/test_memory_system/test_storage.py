"""Tests for InMemoryVectorStore and LocalDiskVectorStore."""

from pathlib import Path
from vajra_agent.memory.storage import InMemoryVectorStore, LocalDiskVectorStore, VectorRecord


def test_in_memory_vector_store():
    store = InMemoryVectorStore()
    rec1 = VectorRecord(text="Python code", vector=[1.0, 0.0, 0.0])
    rec2 = VectorRecord(text="Go code", vector=[0.0, 1.0, 0.0])

    store.add(rec1)
    store.add(rec2)
    assert len(store) == 2

    matches = store.search(query_vector=[1.0, 0.0, 0.0], top_k=1)
    assert len(matches) == 1
    assert matches[0][0].text == "Python code"


def test_local_disk_vector_store(tmp_path: Path):
    file_path = tmp_path / "vecs.json"
    store = LocalDiskVectorStore(file_path)
    rec = VectorRecord(text="Persistent text", vector=[0.5, 0.5])
    store.add(rec)

    # Reload from disk
    store2 = LocalDiskVectorStore(file_path)
    assert len(store2) == 1
    assert store2.search([0.5, 0.5])[0][0].text == "Persistent text"
