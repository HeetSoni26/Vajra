"""MemoryManager central coordinator managing short-term, long-term memory, knowledge graphs, and retrieval."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from vajra_agent.context.manager import ProjectContext
from vajra_agent.indexing.indexer import WorkspaceIndexer
from vajra_agent.memory.context_builder import ContextBuilder
from vajra_agent.memory.embeddings.base import BaseEmbeddingProvider
from vajra_agent.memory.embeddings.mock import MockEmbeddingProvider
from vajra_agent.memory.knowledge.graph import RepositoryKnowledgeGraph
from vajra_agent.memory.policies.base import BaseMemoryPolicy, LRUPolicy
from vajra_agent.memory.retrieval.engine import RetrievalEngine, RetrievalResult
from vajra_agent.memory.storage.base import BaseVectorStore, VectorRecord
from vajra_agent.memory.storage.in_memory import InMemoryVectorStore
from vajra_agent.memory.summaries.summarizer import MemorySummarizer
from vajra_agent.repository.scanner import RepositoryScanner


class MemoryManager:
    """Central Memory Coordinator interface used by FoundationAgent."""

    def __init__(
        self,
        embedder: BaseEmbeddingProvider | None = None,
        vector_store: BaseVectorStore | None = None,
        policy: BaseMemoryPolicy | None = None,
        retrieval_decay_hours: float = 24.0,
    ) -> None:
        self.embedder = embedder or MockEmbeddingProvider()
        self.vector_store = vector_store or InMemoryVectorStore()
        self.policy = policy or LRUPolicy()
        self.retrieval_engine = RetrievalEngine(
            embedder=self.embedder,
            vector_store=self.vector_store,
            recency_decay_hours=retrieval_decay_hours,
        )
        self.knowledge_graph = RepositoryKnowledgeGraph()
        self.summarizer = MemorySummarizer()
        self.project_context: ProjectContext | None = None

    def remember(self, text: str, metadata: dict[str, Any] | None = None) -> VectorRecord:
        """Embed text and persist into vector memory."""
        vec = self.embedder.embed_text(text)
        rec = VectorRecord(text=text, vector=vec, metadata=metadata or {})
        self.vector_store.add(rec)
        return rec

    def recall(
        self,
        query: str,
        top_k: int = 5,
        metadata_filter: dict[str, Any] | None = None,
    ) -> list[RetrievalResult]:
        """Query semantic long-term memory."""
        return self.retrieval_engine.retrieve(query, top_k=top_k, metadata_filter=metadata_filter)

    def index_repository(self, workspace_root: str | Path = ".") -> None:
        """Scan repository structure and index AST symbols into knowledge graph and vector memory."""
        root_path = Path(workspace_root).resolve()
        repo_ctx = RepositoryScanner.scan(root_path)
        index = WorkspaceIndexer.index_directory(root_path)

        self.project_context = ProjectContext(
            workspace_root=str(root_path),
            repo_context=repo_ctx,
            workspace_index=index,
        )

        self.knowledge_graph.build_from_workspace(repo_ctx, index)

        # Index symbols into vector memory for semantic search
        for sym in index.symbols:
            text = f"Symbol {sym.kind} {sym.name} in {sym.filepath}:{sym.line_no}. Doc: {sym.docstring}"
            self.remember(
                text, metadata={"kind": sym.kind, "filepath": sym.filepath, "name": sym.name}
            )

    def build_context(
        self,
        state: Any,
        tool_schemas: list[dict[str, Any]],
        query: str = "",
        custom_system_prompt: str | None = None,
    ) -> tuple[str, str]:
        """Construct full enriched context (sys_prompt, history_prompt) for FoundationAgent."""
        retrieved = self.recall(query, top_k=3) if query else []
        return ContextBuilder.build_enriched_context(
            state=state,
            tool_schemas=tool_schemas,
            project_context=self.project_context,
            knowledge_graph=self.knowledge_graph,
            retrieved_memories=retrieved,
            custom_system_prompt=custom_system_prompt,
        )
