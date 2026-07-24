"""Structured Agent Communication models (Task delegation, progress updates, approvals)."""

from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class MessageType(str, Enum):
    """Structured message types exchanged between agents."""

    TASK_DELEGATION = "task_delegation"
    PROGRESS_UPDATE = "progress_update"
    SHARED_OBSERVATION = "shared_observation"
    QUESTION = "question"
    RESPONSE = "response"
    APPROVAL = "approval"


@dataclass
class AgentMessage:
    """Structured message passed between agents during multi-agent collaboration."""

    sender_id: str
    recipient_id: str
    msg_type: MessageType
    payload: dict[str, Any]
    id: str = field(default_factory=lambda: f"msg_{uuid.uuid4().hex[:8]}")
    timestamp: float = field(default_factory=time.time)
