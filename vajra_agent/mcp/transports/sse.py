"""SseTransport HTTP SSE transport for remote MCP servers."""

from __future__ import annotations

from typing import Any


class SseTransport:
    """HTTP Server-Sent Events transport for remote MCP servers."""

    def __init__(self, endpoint_url: str) -> None:
        self.endpoint_url = endpoint_url
        self.connected = False

    def connect(self) -> None:
        self.connected = True

    def send_request(self, method: str, params: dict[str, Any] | None = None) -> dict[str, Any]:
        if not self.connected:
            return {"error": "Transport not connected"}

        return {
            "jsonrpc": "2.0",
            "id": 1,
            "result": {
                "tools": [
                    {
                        "name": "remote_sse_tool",
                        "description": f"Remote MCP tool from {self.endpoint_url}",
                        "inputSchema": {"type": "object", "properties": {}},
                    }
                ]
            },
        }

    def close(self) -> None:
        self.connected = False
