"""Built-in ShellTool for command execution."""

from __future__ import annotations

import subprocess
from typing import Any

from vajra_agent.tools.base import BaseTool


class ShellTool(BaseTool):
    """Tool for executing shell commands safely with execution timeouts."""

    def __init__(self, timeout: float = 30.0) -> None:
        self.default_timeout = timeout

    @property
    def name(self) -> str:
        return "shell_tool"

    @property
    def description(self) -> str:
        return "Execute shell command lines and return stdout/stderr output."

    @property
    def input_schema(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "command": {
                    "type": "string",
                    "description": "Shell command line string to execute.",
                },
                "timeout": {
                    "type": "number",
                    "description": "Optional timeout in seconds.",
                },
            },
            "required": ["command"],
        }

    def execute(self, command: str, timeout: float | None = None, **kwargs: Any) -> dict[str, Any]:
        t_out = timeout if timeout is not None else self.default_timeout
        try:
            result = subprocess.run(
                command,
                shell=True,
                capture_output=True,
                text=True,
                timeout=t_out,
            )
            return {
                "exit_code": result.returncode,
                "stdout": result.stdout.strip(),
                "stderr": result.stderr.strip(),
            }
        except subprocess.TimeoutExpired:
            return {
                "exit_code": -1,
                "stdout": "",
                "stderr": f"Command timed out after {t_out} seconds.",
            }
