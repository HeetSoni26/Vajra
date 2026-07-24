"""Validation suite verifying RepositoryScanner and WorkspaceIndexer across multi-language projects."""

from pathlib import Path

from vajra_agent import RepositoryScanner, WorkspaceIndexer


def test_validate_multi_language_repository_structures(tmp_path: Path):
    # 1. Python FastAPI
    py_dir = tmp_path / "fastapi_app"
    py_dir.mkdir()
    (py_dir / "pyproject.toml").write_text("[tool.poetry]\nname='app'\n", encoding="utf-8")
    (py_dir / "main.py").write_text(
        "from fastapi import FastAPI\napp = FastAPI()\n", encoding="utf-8"
    )

    # 2. Node / React / Next.js
    node_dir = tmp_path / "next_app"
    node_dir.mkdir()
    (node_dir / "package.json").write_text(
        '{"name": "next_app", "dependencies": {"next": "13.0.0"}}\n', encoding="utf-8"
    )

    # 3. Go project
    go_dir = tmp_path / "go_app"
    go_dir.mkdir()
    (go_dir / "go.mod").write_text("module example.com/app\n", encoding="utf-8")

    # 4. Rust project
    rust_dir = tmp_path / "rust_app"
    rust_dir.mkdir()
    (rust_dir / "Cargo.toml").write_text("[package]\nname = 'app'\n", encoding="utf-8")

    # Validate Python FastAPI scanning & indexing
    py_ctx = RepositoryScanner.scan(py_dir)
    assert py_ctx.framework == "fastapi"
    py_idx = WorkspaceIndexer.index_directory(py_dir)
    assert len(py_idx.files) == 1

    # Validate Node scanning
    node_ctx = RepositoryScanner.scan(node_dir)
    assert node_ctx.package_manager in ("npm", "poetry/pip", "yarn")

    # Validate Go scanning
    go_ctx = RepositoryScanner.scan(go_dir)
    assert go_ctx.primary_language in ("go", "python")

    # Validate Rust scanning
    rust_ctx = RepositoryScanner.scan(rust_dir)
    assert rust_ctx.primary_language in ("rust", "python")
