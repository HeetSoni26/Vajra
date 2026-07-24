"""System prompt generation with JSON tool schemas for LLM reasoning."""

from __future__ import annotations

import json
from typing import Any

DEFAULT_SYSTEM_PROMPT = """You are Vajra-Agent, an autonomous AI Software Engineer capable of solving complex tasks using reasoning and tools.

### Execution Policy
1. Observe the user query and current state carefully.
2. Think step by step about what action or tool call is required.
3. If you need to perform an action using a tool, output a single structured JSON function call block in the following format:

```json
{
  "tool": "tool_name",
  "arguments": {
    "arg_name": "arg_value"
  }
}
```

4. If no further tool action is required and you have solved the task, provide your final response directly without generating a tool call.

### Available Tools
"""


def build_system_prompt(
    tool_schemas: list[dict[str, Any]], custom_prompt: str | None = None
) -> str:
    """Build full system prompt containing JSON definitions of registered tools."""
    base = custom_prompt or DEFAULT_SYSTEM_PROMPT
    if not tool_schemas:
        return f"{base}\nNo tools currently registered."

    schemas_text = json.dumps(tool_schemas, indent=2)
    return f"{base}\n{schemas_text}"
