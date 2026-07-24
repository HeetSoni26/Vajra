"""Abstract Base Tool interface for Vajra-Agent tools."""

from __future__ import annotations

import time
from abc import ABC, abstractmethod
from typing import Any

from vajra_agent.schemas.results import ToolExecutionResult


class BaseTool(ABC):
    """Abstract base class for all tools executable by Vajra-Agent.

    Tools operate independently of any specific LLM and expose standard
    metadata schemas and an execution method.
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """Unique identifier name of the tool."""

    @property
    @abstractmethod
    def description(self) -> str:
        """Description explaining when and how the model should use this tool."""

    @property
    def input_schema(self) -> dict[str, Any]:
        """JSON Schema defining valid input parameters."""
        return {"type": "object", "properties": {}}

    @property
    def output_schema(self) -> dict[str, Any]:
        """JSON Schema defining output shape."""
        return {"type": "object", "properties": {}}

    @abstractmethod
    def execute(self, **kwargs: Any) -> Any:
        """Execute tool logic and return raw result or payload."""

    def run(self, **kwargs: Any) -> ToolExecutionResult:
        """Safely execute the tool and return a strongly typed ToolExecutionResult."""
        start_t = time.perf_counter()
        try:
            output = self.execute(**kwargs)
            duration_ms = round((time.perf_counter() - start_t) * 1000, 2)
            return ToolExecutionResult(
                tool_name=self.name,
                success=True,
                output=output,
                execution_time_ms=duration_ms,
            )
        except Exception as e:
            duration_ms = round((time.perf_counter() - start_t) * 1000, 2)
            return ToolExecutionResult(
                tool_name=self.name,
                success=False,
                output=None,
                error=str(e),
                execution_time_ms=duration_ms,
            )
