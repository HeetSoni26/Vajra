# Tool System & Custom Tool Creation Guide

The Vajra-Agent Tool System enables LLMs to execute actions safely.

## Creating a Custom Tool

Subclass `BaseTool` and define `name`, `description`, `input_schema`, and `execute()`:

```python
from typing import Any
from vajra_agent import BaseTool, FoundationAgent

class WeatherTool(BaseTool):
    @property
    def name(self) -> str:
        return "weather_tool"

    @property
    def description(self) -> str:
        return "Retrieve current weather information for a city."

    @property
    def input_schema(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "location": {"type": "string", "description": "City name"}
            },
            "required": ["location"]
        }

    def execute(self, location: str, **kwargs: Any) -> str:
        return f"Weather in {location} is Sunny, 22°C"
```

## Registering Tools with the Agent

```python
agent = FoundationAgent(reasoner)
agent.register_tool(WeatherTool())
```
