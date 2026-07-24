"""Built-in PythonTool for executing Python code snippets securely."""

from __future__ import annotations

import io
import sys
import traceback
from typing import Any

from vajra_agent.tools.base import BaseTool

BLOCKED_MODULES = {"subprocess", "ctypes", "pty", "socket", "signal", "shutil"}


def _safe_import(
    name: str,
    globals: dict[str, Any] | None = None,
    locals: dict[str, Any] | None = None,
    fromlist: tuple[str, ...] = (),
    level: int = 0,
) -> Any:
    base = name.split(".")[0]
    if base in BLOCKED_MODULES:
        raise ImportError(f"Importing restricted module '{name}' is disallowed for security.")
    return __import__(name, globals, locals, fromlist, level)


def _get_safe_builtins() -> dict[str, Any]:
    import builtins

    safe = {k: getattr(builtins, k) for k in dir(builtins) if not k.startswith("_")}
    # Remove dangerous builtins
    safe.pop("eval", None)
    safe.pop("exec", None)
    safe.pop("compile", None)
    safe.pop("open", None)  # Restrict raw open in generic PythonTool
    safe["__import__"] = _safe_import
    return safe


class PythonTool(BaseTool):
    """Tool for evaluating Python code snippets in a restricted global namespace."""

    @property
    def name(self) -> str:
        return "python_tool"

    @property
    def description(self) -> str:
        return "Execute Python code snippets securely and capture stdout output or errors."

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

        global_namespace: dict[str, Any] = {"__builtins__": _get_safe_builtins()}
        error = None

        try:
            exec(code, global_namespace)
        except Exception as e:
            error = f"{type(e).__name__}: {e!s}\n{traceback.format_exc()}"
        finally:
            sys.stdout = old_stdout

        return {
            "output": buffer.getvalue().strip(),
            "error": error,
            "success": error is None,
        }
