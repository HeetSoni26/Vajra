"""Strongly typed tool execution and agent response models."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class ToolExecutionResult:
    """Strongly typed result model returned by all tool executions."""

    tool_name: str
    success: bool
    output: Any
    error: str | None = None
    execution_time_ms: float = 0.0
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "tool_name": self.tool_name,
            "success": self.success,
            "output": self.output,
            "error": self.error,
            "execution_time_ms": self.execution_time_ms,
            "metadata": self.metadata,
        }


@dataclass
class AgentResponse:
    """Final response returned by FoundationAgent after reasoning and tool execution."""

    output: str
    iterations: int
    tool_calls_count: int
    conversation: list[dict[str, Any]]
    execution_time_s: float
    metadata: dict[str, Any] = field(default_factory=dict)
