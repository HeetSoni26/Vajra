"""Agent Event definitions for logging, telemetry, and observability."""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any


@dataclass
class AgentEvent:
    """Base event emitted by FoundationAgent lifecycle."""

    timestamp: float = field(default_factory=time.time)
    event_name: str = "AgentEvent"


@dataclass
class AgentStarted(AgentEvent):
    prompt: str = ""
    event_name: str = "AgentStarted"


@dataclass
class AgentFinished(AgentEvent):
    output: str = ""
    iterations: int = 0
    event_name: str = "AgentFinished"


@dataclass
class IterationStarted(AgentEvent):
    iteration: int = 0
    event_name: str = "IterationStarted"


@dataclass
class IterationFinished(AgentEvent):
    iteration: int = 0
    has_tool_call: bool = False
    event_name: str = "IterationFinished"


@dataclass
class ToolStarted(AgentEvent):
    tool_name: str = ""
    arguments: dict[str, Any] = field(default_factory=dict)
    event_name: str = "ToolStarted"


@dataclass
class ToolFinished(AgentEvent):
    tool_name: str = ""
    output: Any = None
    execution_time_ms: float = 0.0
    event_name: str = "ToolFinished"


@dataclass
class ToolFailed(AgentEvent):
    tool_name: str = ""
    error: str = ""
    event_name: str = "ToolFailed"
