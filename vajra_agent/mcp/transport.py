"""Abstract Transport interface for MCP protocol communication."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any


class Transport(ABC):
    """Abstract base class for MCP communication transports (e.g. STDIO, SSE)."""

    @abstractmethod
    def connect(self) -> None:
        """Establish transport connection."""

    @abstractmethod
    def disconnect(self) -> None:
        """Close transport connection."""

    @abstractmethod
    def send_message(self, message: dict[str, Any]) -> None:
        """Send JSON-RPC message over transport."""

    @abstractmethod
    def receive_message(self) -> dict[str, Any]:
        """Receive JSON-RPC message from transport."""
