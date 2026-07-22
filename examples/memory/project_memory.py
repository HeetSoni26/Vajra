"""Project memory example attaching project context."""

from vajra_agent import MemoryManager


def main():
    mem = MemoryManager()
    mem.index_repository(".")

    print("Project Memory Overview:")
    print(f"  Framework: {mem.project_context.repo_context.framework}")
    print(f"  Primary Language: {mem.project_context.repo_context.primary_language}")

    recalled = mem.recall("InferenceEngine", top_k=2)
    print("\nRecalled Project Memories for 'InferenceEngine':")
    for r in recalled:
        print(f"  - [{r.final_score:.2f}] {r.record.text}")


if __name__ == "__main__":
    main()
