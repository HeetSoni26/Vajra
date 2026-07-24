"""Built-in ShellTool for command execution with enhanced security."""

from __future__ import annotations

import os
import shlex
import subprocess
from typing import Any

from vajra_agent.tools.base import BaseTool


class ShellTool(BaseTool):
    """Tool for executing shell commands safely with execution timeouts and safe argument parsing."""

    def __init__(self, timeout: float = 30.0) -> None:
        self.default_timeout = timeout

    @property
    def name(self) -> str:
        return "shell_tool"

    @property
    def description(self) -> str:
        return "Execute command lines safely and return stdout/stderr output."

    @property
    def input_schema(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "command": {
                    "type": "string",
                    "description": "Command line string to execute.",
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
        if not command or not command.strip():
            return {"exit_code": 1, "stdout": "", "stderr": "Empty command string provided."}

        # Parse command safely without shell=True
        try:
            if os.name == "nt":
                cmd_args = ["cmd.exe", "/c", command]
            else:
                cmd_args = shlex.split(command)
        except Exception as e:
            return {"exit_code": 1, "stdout": "", "stderr": f"Command parsing error: {e}"}

        if not cmd_args:
            return {
                "exit_code": 1,
                "stdout": "",
                "stderr": "No executable arguments found in command.",
            }

        try:
            result = subprocess.run(
                cmd_args,
                shell=False,
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
        except FileNotFoundError:
            return {
                "exit_code": 127,
                "stdout": "",
                "stderr": f"Executable file not found: {cmd_args[0]}",
            }
        except Exception as e:
            return {
                "exit_code": 1,
                "stdout": "",
                "stderr": f"Execution error: {e}",
            }
