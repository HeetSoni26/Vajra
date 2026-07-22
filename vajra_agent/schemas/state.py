"""Central AgentState maintaining runtime, conversation, planner, and tool execution state."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from vajra_agent.schemas.messages import Conversation
from vajra_agent.schemas.results import ToolExecutionResult


@dataclass
class ExecutionMetadata:
    """Execution statistics and performance metadata."""

    start_time: float = 0.0
    end_time: float = 0.0
    total_tokens_generated: int = 0
    total_tool_calls: int = 0
    errors_encountered: int = 0


@dataclass
class AgentState:
    """Central state object holding the complete runtime context of FoundationAgent."""

    conversation: Conversation = field(default_factory=Conversation)
    current_iteration: int = 0
    max_iterations: int = 10
    tool_history: list[ToolExecutionResult] = field(default_factory=list)
    metadata: ExecutionMetadata = field(default_factory=ExecutionMetadata)
    runtime_context: dict[str, Any] = field(default_factory=dict)
    
    # Placeholders for future phases (Planner, Reflection, RAG)
    plan_steps: list[str] = field(default_factory=list)
    reflection_notes: list[str] = field(default_factory=list)

    def record_tool_result(self, result: ToolExecutionResult) -> None:
        """Record tool execution in central state."""
        self.tool_history.append(result)
        self.metadata.total_tool_calls += 1
        if not result.success:
            self.metadata.errors_encountered += 1
