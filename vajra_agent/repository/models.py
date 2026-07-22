"""Repository context models."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class RepositoryContext:
    """Structured context representation of a codebase project directory."""

    project_root: str
    primary_language: str = "python"
    package_manager: str = "pip"
    framework: str = "generic"
    entry_points: list[str] = field(default_factory=list)
    config_files: list[str] = field(default_factory=list)
    dependency_files: list[str] = field(default_factory=list)
    build_system: str = "setuptools"
    total_files: int = 0
    directory_tree: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "project_root": self.project_root,
            "primary_language": self.primary_language,
            "package_manager": self.package_manager,
            "framework": self.framework,
            "entry_points": self.entry_points,
            "config_files": self.config_files,
            "dependency_files": self.dependency_files,
            "build_system": self.build_system,
            "total_files": self.total_files,
        }
