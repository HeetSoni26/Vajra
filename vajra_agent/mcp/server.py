"""Abstract MCPServer interface for exposing agent tools to external clients."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from vajra_agent.mcp.transport import Transport


class MCPServer(ABC):
    """Abstract interface for exposing Vajra-Agent tools as an MCP server."""

    def __init__(self, transport: Transport) -> None:
        self.transport = transport

    @abstractmethod
    def register_tool_definition(self, tool_def: dict[str, Any]) -> None:
        """Register a tool definition to expose via MCP."""
        pass

    @abstractmethod
    def start(self) -> None:
        """Start listening for incoming MCP requests."""
        pass
