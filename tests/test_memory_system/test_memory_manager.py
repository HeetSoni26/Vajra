"""Tests for MemoryManager remember, recall, and repository indexing."""

from pathlib import Path
from vajra_agent.memory import MemoryManager


def test_memory_manager_remember_and_recall():
    mem = MemoryManager()
    mem.remember("Database schema design using PostgreSQL", metadata={"tag": "db"})
    mem.remember("Frontend UI design using React", metadata={"tag": "ui"})

    results = mem.recall("PostgreSQL", top_k=1)
    assert len(results) == 1
    assert "Database" in results[0].record.text


def test_memory_manager_repository_indexing(tmp_path: Path):
    (tmp_path / "app.py").write_text("class Server:\n    pass\n", encoding="utf-8")

    mem = MemoryManager()
    mem.index_repository(tmp_path)

    assert mem.project_context is not None
    results = mem.recall("Server", top_k=1)
    assert len(results) > 0
