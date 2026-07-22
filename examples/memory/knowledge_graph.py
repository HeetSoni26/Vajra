"""Knowledge graph query example."""

from vajra_agent import MemoryManager


def main():
    mem = MemoryManager()
    mem.index_repository(".")

    deps = mem.knowledge_graph.find_dependencies_of("root")
    print(f"Root project dependencies / files count: {len(deps)}")
    print("Sample Root Files:")
    for d in deps[:5]:
        print(f"  - {d}")


if __name__ == "__main__":
    main()
