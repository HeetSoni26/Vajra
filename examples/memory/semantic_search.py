"""Semantic search demonstration using vector memory."""

from vajra_agent import MemoryManager


def main():
    mem = MemoryManager()
    mem.remember("Implemented KV Cache acceleration in inference/kv_cache.py", metadata={"area": "inference"})
    mem.remember("Configured Distributed Data Parallel (DDP) launch scripts in training/", metadata={"area": "training"})
    mem.remember("Built FastAPI endpoints for completion and tokenization in api/main.py", metadata={"area": "api"})

    query = "Where is KV cache implemented?"
    results = mem.recall(query, top_k=2)

    print(f"Query: '{query}'\nResults:")
    for res in results:
        print(f"  Score: {res.final_score:.4f} | Text: {res.record.text}")


if __name__ == "__main__":
    main()
