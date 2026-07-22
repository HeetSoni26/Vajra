"""Tests for ToolRegistry tool management, schema generation, and plugin hooks."""

from vajra_agent.registry import ToolRegistry
from vajra_agent.tools import FileTool, ShellTool


def test_registry_registration_and_lookup():
    reg = ToolRegistry()
    assert len(reg) == 0

    tool = FileTool()
    reg.register(tool)
    assert len(reg) == 1
    assert "file_tool" in reg
    assert reg.get("file_tool") == tool


def test_registry_schema_generation():
    reg = ToolRegistry()
    reg.register(FileTool())
    reg.register(ShellTool())

    schemas = reg.get_tool_schemas()
    assert len(schemas) == 2
    names = [s["name"] for s in schemas]
    assert "file_tool" in names
    assert "shell_tool" in names


def test_registry_plugin_hook():
    reg = ToolRegistry()
    hook_called = False

    def plugin_hook(r: ToolRegistry):
        nonlocal hook_called
        hook_called = True
        r.register(FileTool())

    reg.register_plugin_hook(plugin_hook)
    assert hook_called
    assert "file_tool" in reg
