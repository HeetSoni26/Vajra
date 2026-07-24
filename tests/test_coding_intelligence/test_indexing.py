"""Tests for WorkspaceIndexer and AST symbol extraction."""

from pathlib import Path

from vajra_agent.indexing import WorkspaceIndexer


def test_workspace_indexer_ast_parsing(tmp_path: Path):
    code = """
class Calculator:
    \"\"\"A simple calculator.\"\"\"
    def add(self, a, b):
        return a + b

def multiply(x, y):
    return x * y
"""
    (tmp_path / "calc.py").write_text(code, encoding="utf-8")

    index = WorkspaceIndexer.index_directory(tmp_path)
    assert len(index.files) == 1

    classes = index.get_classes()
    assert len(classes) == 1
    assert classes[0].name == "Calculator"

    functions = index.get_functions()
    assert any(f.name == "multiply" for f in functions)
