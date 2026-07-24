"""ArtifactManager managing versioned execution artifacts, plans, reports, diffs, and logs."""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class Artifact:
    """Represents a versioned execution artifact."""

    id: str
    kind: str  # plan, code_diff, report, log, test_results
    name: str
    version: int
    content: str
    filepath: str | None = None
    timestamp: float = field(default_factory=time.time)
    metadata: dict[str, Any] = field(default_factory=dict)


class ArtifactManager:
    """Manages versioned artifacts saved to workspace or disk."""

    def __init__(self, artifact_dir: str | Path = "checkpoints/artifacts") -> None:
        self.artifact_dir = Path(artifact_dir)
        self.artifact_dir.mkdir(parents=True, exist_ok=True)
        self.artifacts: dict[str, list[Artifact]] = {}

    def create_artifact(
        self,
        kind: str,
        name: str,
        content: str,
        filepath: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> Artifact:
        """Create a new artifact version."""
        existing = self.artifacts.get(name, [])
        version = len(existing) + 1
        art_id = f"art_{name}_{version}"

        art = Artifact(
            id=art_id,
            kind=kind,
            name=name,
            version=version,
            content=content,
            filepath=filepath,
            metadata=metadata or {},
        )

        if name not in self.artifacts:
            self.artifacts[name] = []
        self.artifacts[name].append(art)

        # Save artifact file
        out_file = self.artifact_dir / f"{name}_v{version}.json"
        out_file.write_text(
            json.dumps(
                {
                    "id": art.id,
                    "kind": art.kind,
                    "name": art.name,
                    "version": art.version,
                    "content": art.content,
                    "filepath": art.filepath,
                    "timestamp": art.timestamp,
                    "metadata": art.metadata,
                },
                indent=2,
            ),
            encoding="utf-8",
        )

        return art

    def get_latest_artifact(self, name: str) -> Artifact | None:
        """Retrieve latest version of named artifact."""
        versions = self.artifacts.get(name, [])
        return versions[-1] if versions else None

    def list_artifacts(self) -> list[Artifact]:
        """List latest version of all tracked artifacts."""
        return [versions[-1] for versions in self.artifacts.values() if versions]
