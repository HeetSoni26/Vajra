"""RepositoryScanner analyzing repository structure, languages, frameworks, and build systems."""

from __future__ import annotations

import json
from pathlib import Path

from vajra_agent.repository.models import RepositoryContext


class RepositoryScanner:
    """Scans project directory and auto-detects language, framework, dependencies, and entrypoints."""

    @classmethod
    def scan(cls, directory_path: str | Path) -> RepositoryContext:
        root = Path(directory_path).resolve()
        if not root.exists() or not root.is_dir():
            raise FileNotFoundError(
                f"Repository path does not exist or is not a directory: {directory_path}"
            )

        files = [f for f in root.rglob("*") if f.is_file() and not cls._should_ignore(f)]
        rel_files = [str(f.relative_to(root)) for f in files]

        lang = cls._detect_language(rel_files)
        pkg_mgr = cls._detect_package_manager(rel_files)
        framework = cls._detect_framework(root, rel_files)
        configs = cls._find_configs(rel_files)
        deps = cls._find_dependencies(rel_files)
        entries = cls._find_entry_points(rel_files)

        dir_tree = sorted(list({str(Path(f).parent) for f in rel_files[:100]}))

        return RepositoryContext(
            project_root=str(root),
            primary_language=lang,
            package_manager=pkg_mgr,
            framework=framework,
            entry_points=entries,
            config_files=configs,
            dependency_files=deps,
            build_system=cls._detect_build_system(rel_files),
            total_files=len(files),
            directory_tree=dir_tree,
        )

    @staticmethod
    def _should_ignore(path: Path) -> bool:
        ignore_dirs = {
            ".git",
            ".pytest_cache",
            ".ruff_cache",
            "__pycache__",
            "node_modules",
            "venv",
            ".venv",
            "dist",
            "build",
        }
        return any(part in ignore_dirs for part in path.parts)

    @staticmethod
    def _detect_language(files: list[str]) -> str:
        ext_counts: dict[str, int] = {}
        for f in files:
            ext = Path(f).suffix.lower()
            if ext in {".py", ".js", ".ts", ".tsx", ".jsx", ".go", ".rs", ".java"}:
                ext_counts[ext] = ext_counts.get(ext, 0) + 1

        if not ext_counts:
            return "python"

        top_ext = max(ext_counts, key=ext_counts.get)
        lang_map = {
            ".py": "python",
            ".js": "javascript",
            ".ts": "typescript",
            ".tsx": "typescript-react",
            ".jsx": "javascript-react",
            ".go": "go",
            ".rs": "rust",
            ".java": "java",
        }
        return lang_map.get(top_ext, "python")

    @staticmethod
    def _detect_package_manager(files: list[str]) -> str:
        if "poetry.lock" in files or "pyproject.toml" in files:
            return "poetry/pip"
        if "yarn.lock" in files:
            return "yarn"
        if "package-lock.json" in files or "package.json" in files:
            return "npm"
        if "Cargo.toml" in files:
            return "cargo"
        if "go.mod" in files:
            return "go modules"
        return "pip"

    @staticmethod
    def _detect_framework(root: Path, files: list[str]) -> str:
        if "next.config.js" in files or "next.config.mjs" in files:
            return "nextjs"
        if "pyproject.toml" in files:
            txt = (root / "pyproject.toml").read_text(encoding="utf-8", errors="ignore").lower()
            if "fastapi" in txt:
                return "fastapi"
            if "django" in txt:
                return "django"
            if "flask" in txt:
                return "flask"
        if "package.json" in files:
            try:
                pkg_data = json.loads(
                    (root / "package.json").read_text(encoding="utf-8", errors="ignore")
                )
                deps = {**pkg_data.get("dependencies", {}), **pkg_data.get("devDependencies", {})}
                if "react" in deps:
                    return "react"
                if "express" in deps:
                    return "express"
            except Exception:
                pass

        # Python default inspection
        for f in files:
            if f.endswith(".py"):
                txt = (root / f).read_text(encoding="utf-8", errors="ignore")
                if "from fastapi import" in txt or "FastAPI(" in txt:
                    return "fastapi"
                if "import flask" in txt or "Flask(__name__)" in txt:
                    return "flask"
        return "generic"

    @staticmethod
    def _find_configs(files: list[str]) -> list[str]:
        known = {
            "pyproject.toml",
            "setup.py",
            "package.json",
            "Cargo.toml",
            "go.mod",
            "Makefile",
            "Dockerfile",
            "configs/config.yaml",
        }
        return [
            f
            for f in files
            if f in known or f.endswith(".toml") or f.endswith(".yaml") or f.endswith(".json")
        ]

    @staticmethod
    def _find_dependencies(files: list[str]) -> list[str]:
        known = {
            "requirements.txt",
            "pyproject.toml",
            "environment.yml",
            "package.json",
            "Cargo.toml",
            "go.mod",
        }
        return [f for f in files if f in known]

    @staticmethod
    def _find_entry_points(files: list[str]) -> list[str]:
        known = {
            "main.py",
            "app.py",
            "index.js",
            "index.ts",
            "src/main.rs",
            "main.go",
            "cli/main.py",
            "api/main.py",
        }
        return [f for f in files if f in known]

    @staticmethod
    def _detect_build_system(files: list[str]) -> str:
        if "pyproject.toml" in files:
            return "setuptools/flit/poetry"
        if "package.json" in files:
            return "npm/vite/webpack"
        if "Cargo.toml" in files:
            return "cargo"
        return "setuptools"
