"""Reflection Engine models."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class ReflectionResult:
    """Structured reflection critique generated after execution iterations."""

    task_success: bool
    reasoning_critique: str = ""
    tool_efficiency: str = ""
    verification_feedback: str = ""
    improvements_suggested: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "task_success": self.task_success,
            "reasoning_critique": self.reasoning_critique,
            "tool_efficiency": self.tool_efficiency,
            "verification_feedback": self.verification_feedback,
            "improvements_suggested": self.improvements_suggested,
        }
