"""SharedMemory workspace accessible to all agents in a MultiAgentEngine team."""

from __future__ import annotations

from typing import Any

from vajra_agent.artifacts.manager import ArtifactManager
from vajra_agent.context.manager import ProjectContext
from vajra_agent.memory.knowledge.graph import RepositoryKnowledgeGraph
from vajra_agent.memory.manager import MemoryManager


class SharedMemory:
    """Shared workspace memory accessible by all active agents."""

    def __init__(self, memory_manager: MemoryManager | None = None) -> None:
        self.memory_manager = memory_manager or MemoryManager()
        self.artifact_manager = ArtifactManager()
        self.shared_plans: list[dict[str, Any]] = []
        self.shared_observations: list[dict[str, Any]] = []

    @property
    def project_context(self) -> ProjectContext | None:
        return self.memory_manager.project_context

    @property
    def knowledge_graph(self) -> RepositoryKnowledgeGraph:
        return self.memory_manager.knowledge_graph

    def publish_observation(self, sender_id: str, text: str) -> None:
        """Publish shared observation to team."""
        obs = {"sender": sender_id, "text": text}
        self.shared_observations.append(obs)
        self.memory_manager.remember(f"[{sender_id}] {text}", metadata={"type": "shared_observation"})
