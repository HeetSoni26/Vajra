"""Specialized Agent base class inheriting from FoundationAgent."""

from __future__ import annotations

from typing import Any

from vajra_agent.agent.agent import FoundationAgent
from vajra_agent.config import AgentConfig
from vajra_agent.reasoners.base import BaseReasoner
from vajra_agent.reasoners.mock import MockReasoner


class SpecializedAgent(FoundationAgent):
    """Base class for domain-specialized agents with customized role, prompt, and tools."""

    role: str = "SpecializedAgent"

    def __init__(
        self,
        reasoner: BaseReasoner | Any | None = None,
        config: AgentConfig | None = None,
        system_prompt: str | None = None,
    ) -> None:
        r = reasoner or MockReasoner([f"[{self.role}] Work complete."])
        cfg = config or AgentConfig()
        if system_prompt:
            cfg.system_prompt_override = system_prompt
        super().__init__(reasoner=r, config=cfg)
