"""FunctionCall model for structured tool invocation requests."""

from __future__ import annotations

from dataclasses import dataclass, field
import uuid
from typing import Any


@dataclass
class FunctionCall:
    """Represents a structured function/tool invocation request made by an LLM."""

    tool_name: str
    arguments: dict[str, Any] = field(default_factory=dict)
    raw_text: str = ""
    call_id: str = field(default_factory=lambda: f"call_{uuid.uuid4().hex[:8]}")

    def to_dict(self) -> dict[str, Any]:
        return {
            "tool": self.tool_name,
            "arguments": self.arguments,
            "call_id": self.call_id,
        }
