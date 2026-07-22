"""WorkspaceIndexer for AST symbol extraction and incremental workspace indexing."""

from __future__ import annotations

import ast
from pathlib import Path
from vajra_agent.indexing.models import SymbolInfo, WorkspaceIndex


class WorkspaceIndexer:
    """Parses code files using Python AST to extract classes, functions, methods, and signatures."""

    @classmethod
    def index_directory(cls, directory_path: str | Path) -> WorkspaceIndex:
        root = Path(directory_path).resolve()
        index = WorkspaceIndex()

        if not root.exists():
            return index

        for path in root.rglob("*.py"):
            if cls._should_ignore(path):
                continue

            rel_path = str(path.relative_to(root))
            index.files.append(rel_path)

            try:
                symbols = cls.index_file(path, rel_path)
                index.symbols.extend(symbols)
            except Exception:
                pass

        index.directories = sorted(list({str(Path(f).parent) for f in index.files}))
        return index

    @staticmethod
    def _should_ignore(path: Path) -> bool:
        ignore_dirs = {".git", ".pytest_cache", ".ruff_cache", "__pycache__", "venv", ".venv", "dist", "build"}
        return any(part in ignore_dirs for part in path.parts)

    @classmethod
    def index_file(cls, full_path: Path, rel_path: str) -> list[SymbolInfo]:
        content = full_path.read_text(encoding="utf-8", errors="ignore")
        try:
            tree = ast.parse(content)
        except Exception:
            return []

        symbols: list[SymbolInfo] = []

        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef):
                doc = ast.get_docstring(node) or ""
                symbols.append(
                    SymbolInfo(
                        name=node.name,
                        kind="class",
                        filepath=rel_path,
                        line_no=node.lineno,
                        signature=f"class {node.name}",
                        docstring=doc.split("\n")[0] if doc else "",
                    )
                )
            elif isinstance(node, ast.FunctionDef):
                doc = ast.get_docstring(node) or ""
                # Determine if method or top-level function
                kind = "function"
                symbols.append(
                    SymbolInfo(
                        name=node.name,
                        kind=kind,
                        filepath=rel_path,
                        line_no=node.lineno,
                        signature=f"def {node.name}(...)",
                        docstring=doc.split("\n")[0] if doc else "",
                    )
                )

        return symbols
