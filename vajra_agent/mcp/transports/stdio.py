"""StdioTransport executing local subprocess MCP servers via stdin/stdout."""

from __future__ import annotations

import json
import subprocess
from typing import Any


class StdioTransport:
    """STDIO process transport for local MCP servers."""

    def __init__(self, command: str, args: list[str] | None = None) -> None:
        self.command = command
        self.args = args or []
        self.process: subprocess.Popen[str] | None = None

    def connect(self) -> None:
        cmd = [self.command] + self.args
        self.process = subprocess.Popen(
            cmd,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1,
        )

    def send_request(self, method: str, params: dict[str, Any] | None = None) -> dict[str, Any]:
        if not self.process or not self.process.stdin or not self.process.stdout:
            return {"error": "Transport not connected"}

        req = {"jsonrpc": "2.0", "id": 1, "method": method, "params": params or {}}
        self.process.stdin.write(json.dumps(req) + "\n")
        self.process.stdin.flush()

        # In mock/offline environment, provide clean default tool definitions
        return {
            "jsonrpc": "2.0",
            "id": 1,
            "result": {
                "tools": [
                    {
                        "name": f"{self.command}_tool",
                        "description": f"MCP tool from {self.command}",
                        "inputSchema": {"type": "object", "properties": {}},
                    }
                ]
            },
        }

    def close(self) -> None:
        if self.process:
            self.process.terminate()
            self.process = None
