"""Repository memory scanning example."""

from vajra_agent import MemoryManager


def main():
    mem = MemoryManager()
    mem.index_repository(".")

    kg = mem.knowledge_graph
    summary = kg.to_summary_dict()

    print("Repository Knowledge Summary:")
    print(f"  Total Nodes: {summary['total_nodes']}")
    print(f"  Total Edges: {summary['total_edges']}")
    print(f"  Files Indexed: {summary['files_count']}")
    print(f"  Symbols (Classes/Functions): {summary['symbols_count']}")


if __name__ == "__main__":
    main()
