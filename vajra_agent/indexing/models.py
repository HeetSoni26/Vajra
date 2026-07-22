"""Workspace Index models and searchable symbol structures."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class SymbolInfo:
    """Represents a code symbol (class, function, method, variable)."""

    name: str
    kind: str  # class, function, method, import
    filepath: str
    line_no: int
    signature: str = ""
    docstring: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "kind": self.kind,
            "filepath": self.filepath,
            "line_no": self.line_no,
            "signature": self.signature,
            "docstring": self.docstring,
        }


@dataclass
class WorkspaceIndex:
    """Searchable in-memory index of code entities across a workspace."""

    files: list[str] = field(default_factory=list)
    directories: list[str] = field(default_factory=list)
    symbols: list[SymbolInfo] = field(default_factory=list)

    def find_symbol(self, name: str) -> list[SymbolInfo]:
        """Find symbols matching a given query name."""
        return [s for s in self.symbols if name.lower() in s.name.lower()]

    def get_classes(self) -> list[SymbolInfo]:
        return [s for s in self.symbols if s.kind == "class"]

    def get_functions(self) -> list[SymbolInfo]:
        return [s for s in self.symbols if s.kind in ("function", "method")]

    def search_by_file(self, filepath: str) -> list[SymbolInfo]:
        """Return all symbols defined inside a specific file."""
        return [s for s in self.symbols if filepath in s.filepath]

    def to_dict(self) -> dict[str, Any]:
        return {
            "total_files": len(self.files),
            "total_symbols": len(self.symbols),
            "classes_count": len([s for s in self.symbols if s.kind == "class"]),
            "functions_count": len([s for s in self.symbols if s.kind in ("function", "method")]),
        }
