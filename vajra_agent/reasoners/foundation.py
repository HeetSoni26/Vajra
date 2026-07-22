"""FoundationReasoner wrapping FoundationLM InferenceEngine."""

from __future__ import annotations

from typing import Any

from vajra_agent.reasoners.base import BaseReasoner


class FoundationReasoner(BaseReasoner):
    """Reasoner implementation wrapping FoundationLM InferenceEngine."""

    def __init__(self, engine: Any, max_new_tokens: int = 256, temperature: float = 0.7) -> None:
        self.engine = engine
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature

    def generate(self, prompt: str, system_prompt: str | None = None) -> str:
        from inference.engine import GenerationConfig

        full_prompt = prompt
        if system_prompt:
            full_prompt = f"<|SYSTEM|>\n{system_prompt}\n</|SYSTEM|>\n\n{prompt}"

        gen_cfg = GenerationConfig(
            max_new_tokens=self.max_new_tokens,
            temperature=self.temperature,
            do_sample=(self.temperature > 0),
        )

        results = self.engine.generate(full_prompt, gen_cfg)
        return results[0]
