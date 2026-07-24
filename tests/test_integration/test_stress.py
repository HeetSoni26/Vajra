"""Stress tests verifying system performance under large file counts and deep DAG task graphs."""

from pathlib import Path
from vajra_agent import MemoryManager, MultiAgentEngine, TaskGraph


def test_stress_large_repository_indexing(tmp_path: Path):
    # Generate 50 mock files
    for i in range(50):
        (tmp_path / f"module_{i}.py").write_text(
            f"class Class_{i}:\n    def method_{i}(self):\n        pass\n", encoding="utf-8"
        )

    mem = MemoryManager()
    mem.index_repository(tmp_path)

    summary = mem.knowledge_graph.to_summary_dict()
    assert summary["files_count"] == 50
    assert summary["symbols_count"] >= 50


def test_stress_deep_task_graph_execution():
    engine = MultiAgentEngine()
    engine.setup_default_team()

    graph = TaskGraph()
    prev_id = None
    for i in range(10):
        node = graph.add_task(
            description=f"Step {i} task execution",
            agent_role="CoderAgent",
            dependencies=[prev_id] if prev_id else [],
        )
        prev_id = node.id

    res = engine.run("Deep DAG execution", task_graph=graph)
    assert res.iterations == 10
