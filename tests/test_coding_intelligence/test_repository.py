"""Tests for RepositoryScanner and RepositoryContext."""

from pathlib import Path
from vajra_agent.repository import RepositoryScanner


def test_repository_scanner(tmp_path: Path):
    # Setup dummy project files
    (tmp_path / "pyproject.toml").write_text("[project]\nname='test'", encoding="utf-8")
    (tmp_path / "main.py").write_text("from fastapi import FastAPI\napp = FastAPI()", encoding="utf-8")

    ctx = RepositoryScanner.scan(tmp_path)
    assert ctx.primary_language == "python"
    assert ctx.package_manager == "poetry/pip"
    assert ctx.framework == "fastapi"
    assert "main.py" in ctx.entry_points
    assert "pyproject.toml" in ctx.config_files
