"""Tests for RepositoryKnowledgeGraph."""

from vajra_agent.indexing import SymbolInfo, WorkspaceIndex
from vajra_agent.memory.knowledge import RepositoryKnowledgeGraph
from vajra_agent.repository import RepositoryContext


def test_repository_knowledge_graph_construction():
    graph = RepositoryKnowledgeGraph()
    repo_ctx = RepositoryContext(project_root=".", framework="fastapi")

    index = WorkspaceIndex(
        files=["main.py"],
        symbols=[SymbolInfo(name="app", kind="variable", filepath="main.py", line_no=1)],
    )

    graph.build_from_workspace(repo_ctx, index)
    summary = graph.to_summary_dict()

    assert summary["files_count"] == 1
    assert summary["symbols_count"] == 1
    assert "main.py" in graph.nodes
