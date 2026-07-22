"""Tests for built-in tools (FileTool, ShellTool, PythonTool, GitTool)."""

from pathlib import Path

from vajra_agent.tools import FileTool, GitTool, PythonTool, ShellTool


def test_file_tool_read_write_list(tmp_path: Path):
    tool = FileTool()
    target = tmp_path / "test.txt"

    # Write
    res_write = tool.run(action="write", path=str(target), content="Hello agent!")
    assert res_write.success
    assert target.exists()

    # Read
    res_read = tool.run(action="read", path=str(target))
    assert res_read.success
    assert res_read.output == "Hello agent!"

    # List
    res_list = tool.run(action="list", path=str(tmp_path))
    assert res_list.success
    assert "test.txt" in res_list.output


def test_shell_tool():
    tool = ShellTool()
    res = tool.run(command="echo 'Hello shell'")
    assert res.success
    assert "Hello shell" in res.output["stdout"]


def test_python_tool():
    tool = PythonTool()
    code = "print(2 + 2)"
    res = tool.run(code=code)
    assert res.success
    assert res.output["output"] == "4"


def test_git_tool():
    tool = GitTool()
    res = tool.run(action="status")
    assert res.success
    assert "action" in res.output
