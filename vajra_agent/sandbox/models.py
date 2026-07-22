"""Code execution sandbox models."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class SandboxResult:
    """Detailed execution result returned by PythonSandbox execution."""

    success: bool
    stdout: str = ""
    stderr: str = ""
    exit_code: int = 0
    execution_time_ms: float = 0.0
    files_generated: list[str] = field(default_factory=list)
    error_type: str | None = None
    traceback: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "success": self.success,
            "stdout": self.stdout,
            "stderr": self.stderr,
            "exit_code": self.exit_code,
            "execution_time_ms": self.execution_time_ms,
            "files_generated": self.files_generated,
            "error_type": self.error_type,
            "traceback": self.traceback,
        }
