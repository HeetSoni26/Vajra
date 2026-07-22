"""Tool calling example using FileTool."""

from vajra_agent import FileTool, FoundationAgent, MockReasoner


def main():
    # Mock reasoner requesting FileTool write, then providing final answer
    responses = [
        '```json\n{"tool": "file_tool", "arguments": {"action": "write", "path": "test_output.txt", "content": "Vajra-Agent in action!"}}\n```',
        "I have written 'Vajra-Agent in action!' to test_output.txt.",
    ]
    reasoner = MockReasoner(responses)

    agent = FoundationAgent(reasoner)
    agent.register_tool(FileTool())

    response = agent.run("Create a file called test_output.txt with content 'Vajra-Agent in action!'")
    print("Agent Response:", response.output)
    print("Tool Calls Count:", response.tool_calls_count)


if __name__ == "__main__":
    main()
