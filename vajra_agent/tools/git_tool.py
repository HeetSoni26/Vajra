"""Built-in GitTool for git operations."""

from __future__ import annotations

import subprocess
from typing import Any

from vajra_agent.tools.base import BaseTool


class GitTool(BaseTool):
    """Tool for git operations (status, diff, log)."""

    @property
    def name(self) -> str:
        return "git_tool"

    @property
    def description(self) -> str:
        return "Inspect git status, retrieve git diffs, or query git log history."

    @property
    def input_schema(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "action": {
                    "type": "string",
                    "enum": ["status", "diff", "log"],
                    "description": "Git subcommand action to run.",
                },
                "max_count": {
                    "type": "integer",
                    "description": "Limit for git log history entries.",
                },
            },
            "required": ["action"],
        }

    def execute(self, action: str, max_count: int = 5, **kwargs: Any) -> dict[str, Any]:
        if action == "status":
            cmd = ["git", "status", "--short"]
        elif action == "diff":
            cmd = ["git", "diff"]
        elif action == "log":
            cmd = ["git", "log", f"-n{max_count}", "--oneline"]
        else:
            raise ValueError(f"Unsupported git action: {action}")

        try:
            res = subprocess.run(cmd, capture_output=True, text=True, check=True)
            return {"output": res.stdout.strip(), "action": action}
        except Exception as e:
            return {"error": str(e), "action": action}
