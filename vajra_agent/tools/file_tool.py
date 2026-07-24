"""Built-in FileTool for filesystem operations (read, write, list)."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from vajra_agent.tools.base import BaseTool


class FileTool(BaseTool):
    """Tool for reading, writing, and listing files in the workspace."""

    @property
    def name(self) -> str:
        return "file_tool"

    @property
    def description(self) -> str:
        return "Inspect, read, write, or list files and directories in the filesystem."

    @property
    def input_schema(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "action": {
                    "type": "string",
                    "enum": ["read", "write", "list", "exists"],
                    "description": "Filesystem operation to perform.",
                },
                "path": {
                    "type": "string",
                    "description": "Target file or directory path.",
                },
                "content": {
                    "type": "string",
                    "description": "Content to write when action is 'write'.",
                },
            },
            "required": ["action", "path"],
        }

    def execute(self, action: str, path: str, content: str | None = None, **kwargs: Any) -> Any:
        target = Path(path)

        if action == "read":
            if not target.exists():
                raise FileNotFoundError(f"File not found: {path}")
            return target.read_text(encoding="utf-8")

        elif action == "write":
            if content is None:
                raise ValueError("Content parameter is required for write action")
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(content, encoding="utf-8")
            return f"Successfully wrote {len(content)} characters to {path}"

        elif action == "list":
            if not target.exists():
                raise FileNotFoundError(f"Directory not found: {path}")
            if not target.is_dir():
                raise NotADirectoryError(f"Path is not a directory: {path}")
            return [f.name for f in target.iterdir()]

        elif action == "exists":
            return {
                "exists": target.exists(),
                "is_file": target.is_file(),
                "is_dir": target.is_dir(),
            }

        else:
            raise ValueError(f"Unknown file action: {action}")
