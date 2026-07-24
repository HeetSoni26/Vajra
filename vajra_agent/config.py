"""AgentConfig model defining FoundationAgent execution policies and profiles."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Any


@dataclass
class AgentConfig:
    """Configuration parameters controlling FoundationAgent behavior."""

    max_iterations: int = 10
    tool_timeout: float = 30.0
    verbose: bool = True
    stop_on_error: bool = False
    allow_parallel_tools: bool = False
    system_prompt_override: str | None = None
    runtime_metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Load environment variable overrides if present."""
        if "FA_MAX_ITERATIONS" in os.environ:
            self.max_iterations = int(os.environ["FA_MAX_ITERATIONS"])
        if "FA_VERBOSE" in os.environ:
            self.verbose = os.environ["FA_VERBOSE"].lower() in ("true", "1", "yes")

    @classmethod
    def development(cls) -> AgentConfig:
        """Verbose development configuration profile."""
        return cls(max_iterations=15, verbose=True, stop_on_error=False)

    @classmethod
    def testing(cls) -> AgentConfig:
        """Strict testing configuration profile."""
        return cls(max_iterations=5, verbose=False, stop_on_error=True)

    @classmethod
    def production(cls) -> AgentConfig:
        """High-performance production configuration profile."""
        return cls(max_iterations=20, verbose=False, stop_on_error=False, allow_parallel_tools=True)
