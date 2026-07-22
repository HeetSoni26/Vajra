"""Built-in PythonTool for executing Python code snippets."""

from __future__ import annotations

import io
import sys
import traceback
from typing import Any

from vajra_agent.tools.base import BaseTool


class PythonTool(BaseTool):
    """Tool for evaluating Python code snippets and capturing printed output."""

    @property
    def name(self) -> str:
        return "python_tool"

    @property
    def description(self) -> str:
        return "Execute Python code snippets and capture stdout output or errors."

    @property
    def input_schema(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "code": {
                    "type": "string",
                    "description": "Python code snippet to execute.",
                },
            },
            "required": ["code"],
        }

    def execute(self, code: str, **kwargs: Any) -> dict[str, Any]:
        buffer = io.StringIO()
        old_stdout = sys.stdout
        sys.stdout = buffer

        global_namespace: dict[str, Any] = {}
        error = None

        try:
            exec(code, global_namespace)
        except Exception as e:
            error = f"{type(e).__name__}: {str(e)}\n{traceback.format_exc()}"
        finally:
            sys.stdout = old_stdout

        return {
            "output": buffer.getvalue().strip(),
            "error": error,
            "success": error is None,
        }
