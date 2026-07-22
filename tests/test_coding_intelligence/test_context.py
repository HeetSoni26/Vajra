"""Tests for ProjectContext manager."""

from pathlib import Path
from vajra_agent.context import ProjectContext


def test_project_context_load(tmp_path: Path):
    (tmp_path / "pyproject.toml").write_text("[project]\nname='ctx_test'", encoding="utf-8")
    ctx = ProjectContext.load(tmp_path)

    assert ctx.workspace_root == str(tmp_path.resolve())
    assert ctx.repo_context.package_manager == "poetry/pip"
    assert ctx.to_dict()["active_branch"] == "main"
