"""Abstract BaseReasoner interface for decoupling Vajra-Agent from specific LLMs."""

from __future__ import annotations

from abc import ABC, abstractmethod


class BaseReasoner(ABC):
    """Abstract interface for LLM reasoning engines.

    Allows FoundationAgent to depend on an abstraction rather than a specific
    model framework. Subclasses support Vajra-LM, Hugging Face, Ollama, OpenAI, etc.
    """

    @abstractmethod
    def generate(self, prompt: str, system_prompt: str | None = None) -> str:
        """Generate reasoning output for a given prompt and optional system prompt."""
        pass
