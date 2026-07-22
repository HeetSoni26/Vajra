"""ProjectContext manager persisting workspace metadata, frameworks, and coding standards."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from vajra_agent.indexing.models import WorkspaceIndex
from vajra_agent.repository.models import RepositoryContext
from vajra_agent.repository.scanner import RepositoryScanner


@dataclass
class ProjectContext:
    """Consolidated project context for Vajra-Agent execution."""

    workspace_root: str
    repo_context: RepositoryContext = field(
        default_factory=lambda: RepositoryContext(project_root=".")
    )
    workspace_index: WorkspaceIndex = field(default_factory=WorkspaceIndex)
    recent_changes: list[str] = field(default_factory=list)
    active_branch: str = "main"

    @classmethod
    def load(cls, workspace_root: str | Path = ".") -> ProjectContext:
        root_path = Path(workspace_root).resolve()
        repo_ctx = RepositoryScanner.scan(root_path)
        return cls(
            workspace_root=str(root_path),
            repo_context=repo_ctx,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "workspace_root": self.workspace_root,
            "repository": self.repo_context.to_dict(),
            "active_branch": self.active_branch,
            "recent_changes_count": len(self.recent_changes),
        }
