"""BaseWorkflow interface for specialized coding tasks."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from vajra_agent.agent.agent import FoundationAgent
from vajra_agent.schemas.results import AgentResponse


class BaseWorkflow(ABC):
    """Abstract base class for high-level specialized software engineering workflows."""

    def __init__(self, agent: FoundationAgent) -> None:
        self.agent = agent

    @property
    @abstractmethod
    def name(self) -> str:
        """Workflow identifier name."""

    @abstractmethod
    def execute(self, **kwargs: Any) -> AgentResponse:
        """Execute workflow logic."""
