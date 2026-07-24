"""Custom Tool creation example."""

from typing import Any

from vajra_agent import BaseTool, FoundationAgent, MockReasoner


class CalculatorTool(BaseTool):
    """Custom math calculator tool."""

    @property
    def name(self) -> str:
        return "calculator"

    @property
    def description(self) -> str:
        return "Perform basic arithmetic calculations (add, subtract, multiply, divide)."

    @property
    def input_schema(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "op": {"type": "string", "enum": ["add", "sub", "mul", "div"]},
                "a": {"type": "number"},
                "b": {"type": "number"},
            },
            "required": ["op", "a", "b"],
        }

    def execute(self, op: str, a: float, b: float, **kwargs: Any) -> float:
        if op == "add":
            return a + b
        elif op == "sub":
            return a - b
        elif op == "mul":
            return a * b
        elif op == "div":
            if b == 0:
                raise ValueError("Division by zero")
            return a / b
        raise ValueError(f"Unknown operation {op}")


def main():
    responses = [
        '```json\n{"tool": "calculator", "arguments": {"op": "mul", "a": 7, "b": 6}}\n```',
        "The result of 7 * 6 is 42.",
    ]
    reasoner = MockReasoner(responses)

    agent = FoundationAgent(reasoner)
    agent.register_tool(CalculatorTool())

    response = agent.run("Multiply 7 by 6")
    print("Agent Output:", response.output)


if __name__ == "__main__":
    main()
