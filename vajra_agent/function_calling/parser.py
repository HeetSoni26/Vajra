"""Robust JSON extractor, syntax repair, and FunctionCall parser."""

from __future__ import annotations

import json
import re

from vajra_agent.function_calling.call import FunctionCall


def _repair_json_string(text: str) -> str:
    """Repair common minor JSON syntax errors like trailing commas."""
    # Remove trailing commas before closing braces/brackets
    text = re.sub(r",\s*([\}\]])", r"\1", text)
    return text


class FunctionParser:
    """Parses LLM generation text to extract structured FunctionCall requests."""

    @staticmethod
    def extract_json_block(text: str) -> str | None:
        """Extract JSON substring from markdown code blocks or raw JSON string."""
        # 1. Match ```json ... ``` code blocks
        match = re.search(r"```(?:json)?\s*(\{.*\})\s*```", text, re.DOTALL)
        if match:
            return match.group(1).strip()

        # 2. Match outer JSON object boundaries
        start_idx = text.find("{")
        end_idx = text.rfind("}")
        if start_idx != -1 and end_idx != -1 and end_idx > start_idx:
            return text[start_idx : end_idx + 1].strip()

        return None

    @classmethod
    def parse(cls, text: str) -> FunctionCall | None:
        """Extract and parse a FunctionCall from LLM generated text.

        Returns None if no function call request is present or parsing fails.
        """
        raw_json_str = cls.extract_json_block(text)
        if not raw_json_str:
            return None

        repaired_str = _repair_json_string(raw_json_str)

        try:
            data = json.loads(repaired_str)
        except json.JSONDecodeError:
            return None

        if not isinstance(data, dict):
            return None

        # Check for standard tool call keys
        tool_name = data.get("tool") or data.get("name") or data.get("tool_name")
        if not tool_name or not isinstance(tool_name, str):
            return None

        args = data.get("arguments") or data.get("args") or data.get("parameters") or {}
        if not isinstance(args, dict):
            args = {}

        return FunctionCall(
            tool_name=tool_name,
            arguments=args,
            raw_text=text,
        )
