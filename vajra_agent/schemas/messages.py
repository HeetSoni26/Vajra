"""Structured message and conversation models for Vajra-Agent."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class MessageRole(str, Enum):
    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"
    TOOL = "tool"


@dataclass
class Message:
    """Represents a single message in an agent conversation."""

    role: MessageRole | str
    content: str
    name: str | None = None
    tool_call: dict[str, Any] | None = None
    tool_call_id: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert message to dictionary format."""
        d: dict[str, Any] = {
            "role": str(self.role),
            "content": self.content,
        }
        if self.name:
            d["name"] = self.name
        if self.tool_call:
            d["tool_call"] = self.tool_call
        if self.tool_call_id:
            d["tool_call_id"] = self.tool_call_id
        if self.metadata:
            d["metadata"] = self.metadata
        return d


@dataclass
class Conversation:
    """Manages ordered sequence of messages in an agent session."""

    messages: list[Message] = field(default_factory=list)

    def add_message(self, message: Message) -> None:
        """Add a message to the conversation."""
        self.messages.append(message)

    def add_system(self, content: str) -> None:
        """Helper to add a system message."""
        self.add_message(Message(role=MessageRole.SYSTEM, content=content))

    def add_user(self, content: str) -> None:
        """Helper to add a user message."""
        self.add_message(Message(role=MessageRole.USER, content=content))

    def add_assistant(self, content: str, tool_call: dict[str, Any] | None = None) -> None:
        """Helper to add an assistant message."""
        self.add_message(Message(role=MessageRole.ASSISTANT, content=content, tool_call=tool_call))

    def add_tool_result(self, tool_name: str, content: str, tool_call_id: str | None = None) -> None:
        """Helper to add a tool execution result message."""
        self.add_message(
            Message(role=MessageRole.TOOL, content=content, name=tool_name, tool_call_id=tool_call_id)
        )

    def to_list(self) -> list[dict[str, Any]]:
        """Return list of dictionary message objects."""
        return [m.to_dict() for m in self.messages]

    def format_history_string(self) -> str:
        """Format conversation into a single string for legacy prompt-based LLMs."""
        formatted = []
        for msg in self.messages:
            role_str = str(msg.role).upper()
            formatted.append(f"<{role_str}>\n{msg.content}\n</{role_str}>")
        return "\n\n".join(formatted)
