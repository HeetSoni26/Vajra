"""Multiple tools orchestration example."""

from vajra_agent import (
    FileTool,
    FoundationAgent,
    GitTool,
    MockReasoner,
    PythonTool,
    ShellTool,
)


def main():
    responses = [
        '```json\n{"tool": "python_tool", "arguments": {"code": "print(10 + 20)"}}\n```',
        '```json\n{"tool": "file_tool", "arguments": {"action": "write", "path": "calc.txt", "content": "Result: 30"}}\n```',
        "Calculated 10 + 20 = 30 and saved result to calc.txt.",
    ]
    reasoner = MockReasoner(responses)

    agent = FoundationAgent(reasoner)
    agent.register_tool(PythonTool())
    agent.register_tool(FileTool())
    agent.register_tool(ShellTool())
    agent.register_tool(GitTool())

    response = agent.run("Calculate 10 + 20 then save to calc.txt.")
    print("Final Output:", response.output)
    print("Total Tool Executions:", response.tool_calls_count)


if __name__ == "__main__":
    main()
