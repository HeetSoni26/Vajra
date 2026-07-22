"""Persistent session memory example using LocalDiskVectorStore."""

from pathlib import Path
from vajra_agent import LocalDiskVectorStore, MemoryManager, MockEmbeddingProvider


def main():
    store_path = Path("checkpoints/agent_memory/session_store.json")
    embedder = MockEmbeddingProvider()
    disk_store = LocalDiskVectorStore(store_path)

    mem = MemoryManager(embedder=embedder, vector_store=disk_store)
    mem.remember("Architectural decision: Use safetensors format for model export.")

    print(f"Session memory persisted to {store_path}")
    print(f"Stored records count: {len(disk_store)}")


if __name__ == "__main__":
    main()
