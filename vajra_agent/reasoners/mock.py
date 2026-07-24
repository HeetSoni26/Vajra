"""MockReasoner for testing and deterministic agent loop verification."""

from __future__ import annotations

from collections.abc import Callable

from vajra_agent.reasoners.base import BaseReasoner


class MockReasoner(BaseReasoner):
    """Mock reasoner returning pre-scripted responses or dynamic responses via callback."""

    def __init__(self, responses: list[str] | Callable[[str], str] | None = None) -> None:
        self.responses = responses or []
        self._call_count = 0

    def generate(self, prompt: str, system_prompt: str | None = None) -> str:
        self._call_count += 1
        if callable(self.responses):
            return self.responses(prompt)
        elif isinstance(self.responses, list) and self.responses:
            idx = min(self._call_count - 1, len(self.responses) - 1)
            return self.responses[idx]
        return "I am a mock response."
