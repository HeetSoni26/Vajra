"""Function calling parsing and validation example."""

from vajra_agent.function_calling import FunctionParser


def main():
    generation_output = """
I will run the calculation in Python.

```json
{
  "tool": "python_tool",
  "arguments": {
    "code": "import math; print(math.factorial(5))"
  }
}
```
"""

    call = FunctionParser.parse(generation_output)
    if call:
        print("Parsed Function Call:")
        print(f"  Tool Name: {call.tool_name}")
        print(f"  Arguments: {call.arguments}")
        print(f"  Call ID: {call.call_id}")
    else:
        print("No function call found.")


if __name__ == "__main__":
    main()
