"""Abstract MCPClient interface for interacting with external MCP servers."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from vajra_agent.mcp.transport import Transport


class MCPClient(ABC):
    """Abstract interface for Model Context Protocol client."""

    def __init__(self, transport: Transport) -> None:
        self.transport = transport

    @abstractmethod
    def list_tools(self) -> list[dict[str, Any]]:
        """List available tools exposed by the MCP server."""
        pass

    @abstractmethod
    def call_tool(self, name: str, arguments: dict[str, Any]) -> Any:
        """Call a specific tool on the MCP server."""
        pass
