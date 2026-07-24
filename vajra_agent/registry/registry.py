"""Central ToolRegistry for tool lookup, validation, and future plugin discovery."""

from __future__ import annotations

from typing import Any, Callable

from vajra_agent.tools.base import BaseTool

PluginHook = Callable[["ToolRegistry"], None]


class ToolRegistry:
    """Central registry of executable tools.

    Acts as the single source of truth for available agent tools and schema generation.
    Supports plugin hook registration for future extensibility.
    """

    def __init__(self) -> None:
        self._tools: dict[str, BaseTool] = {}
        self._plugin_hooks: list[PluginHook] = []

    def register(self, tool: BaseTool) -> None:
        """Register a new tool instance."""
        if not isinstance(tool, BaseTool):
            raise TypeError(f"Expected BaseTool instance, got {type(tool)}")
        self._tools[tool.name] = tool

    def unregister(self, name: str) -> None:
        """Remove a tool from the registry."""
        if name in self._tools:
            del self._tools[name]

    def get(self, name: str) -> BaseTool | None:
        """Look up a tool by name."""
        return self._tools.get(name)

    def list_tools(self) -> list[BaseTool]:
        """Return list of all registered tools."""
        return list(self._tools.values())

    def get_tool_schemas(self) -> list[dict[str, Any]]:
        """Generate JSON Schema definitions for all registered tools."""
        schemas = []
        for tool in self._tools.values():
            schemas.append(
                {
                    "name": tool.name,
                    "description": tool.description,
                    "parameters": tool.input_schema,
                }
            )
        return schemas

    def register_plugin_hook(self, hook: PluginHook) -> None:
        """Register a plugin discovery hook for future phase extensions."""
        self._plugin_hooks.append(hook)
        # Execute hook immediately
        hook(self)

    def __len__(self) -> int:
        return len(self._tools)

    def __contains__(self, name: str) -> bool:
        return name in self._tools
