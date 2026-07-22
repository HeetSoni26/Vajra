"""RepositoryKnowledgeGraph mapping architectural relationships, dependencies, and symbols."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from vajra_agent.indexing.models import WorkspaceIndex
from vajra_agent.repository.models import RepositoryContext


@dataclass
class KnowledgeNode:
    """Represents a node in the codebase knowledge graph (file, class, function)."""

    id: str
    kind: str  # file, class, function, dependency
    name: str
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class KnowledgeEdge:
    """Represents a directional relationship between nodes (imports, contains, calls)."""

    source_id: str
    target_id: str
    relationship: str  # contains, imports, depends_on


class RepositoryKnowledgeGraph:
    """Graph mapping codebase structural relationships, classes, functions, and imports."""

    def __init__(self) -> None:
        self.nodes: dict[str, KnowledgeNode] = {}
        self.edges: list[KnowledgeEdge] = []

    def add_node(self, node: KnowledgeNode) -> None:
        self.nodes[node.id] = node

    def add_edge(self, source_id: str, target_id: str, relationship: str) -> None:
        self.edges.append(KnowledgeEdge(source_id=source_id, target_id=target_id, relationship=relationship))

    def build_from_workspace(self, repo_ctx: RepositoryContext, index: WorkspaceIndex) -> None:
        """Populate graph nodes and edges from RepositoryContext and WorkspaceIndex."""
        # Add root project node
        root_node = KnowledgeNode(id="root", kind="project", name=repo_ctx.framework)
        self.add_node(root_node)

        # Add files & symbols
        for filepath in index.files:
            file_node = KnowledgeNode(id=filepath, kind="file", name=filepath)
            self.add_node(file_node)
            self.add_edge("root", filepath, "contains")

        for sym in index.symbols:
            sym_id = f"{sym.filepath}::{sym.name}"
            sym_node = KnowledgeNode(id=sym_id, kind=sym.kind, name=sym.name, metadata=sym.to_dict())
            self.add_node(sym_node)
            self.add_edge(sym.filepath, sym_id, "defines")

    def find_dependencies_of(self, node_id: str) -> list[str]:
        """Find target node IDs that node_id depends on or defines."""
        return [edge.target_id for edge in self.edges if edge.source_id == node_id]

    def to_summary_dict(self) -> dict[str, Any]:
        return {
            "total_nodes": len(self.nodes),
            "total_edges": len(self.edges),
            "files_count": len([n for n in self.nodes.values() if n.kind == "file"]),
            "symbols_count": len([n for n in self.nodes.values() if n.kind not in ("project", "file")]),
        }
