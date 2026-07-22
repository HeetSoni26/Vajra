"""ToolAdapter wrapping external MCP tools as native BaseTool instances."""

from __future__ import annotations

from typing import Any

from vajra_agent.mcp.client import MCPClient
from vajra_agent.tools.base import BaseTool


class ToolAdapter(BaseTool):
    """Adapter wrapping an external MCP tool definition as a native BaseTool."""

    def __init__(self, mcp_client: MCPClient, tool_info: dict[str, Any]) -> None:
        self.client = mcp_client
        self._name = tool_info["name"]
        self._description = tool_info.get("description", "")
        self._schema = tool_info.get("parameters", {"type": "object", "properties": {}})

    @property
    def name(self) -> str:
        return self._name

    @property
    def description(self) -> str:
        return self._description

    @property
    def input_schema(self) -> dict[str, Any]:
        return self._schema

    def execute(self, **kwargs: Any) -> Any:
        return self.client.call_tool(self.name, kwargs)
