"""End-to-end unit tests for FoundationAgent execution loop, events, and recursion caps."""

from vajra_agent import AgentConfig, FoundationAgent, MockReasoner
from vajra_agent.tools import PythonTool


def test_agent_single_turn_no_tools():
    reasoner = MockReasoner(["The capital of France is Paris."])
    agent = FoundationAgent(reasoner)

    response = agent.run("What is the capital of France?")
    assert response.output == "The capital of France is Paris."
    assert response.iterations == 1
    assert response.tool_calls_count == 0


def test_agent_tool_calling_loop():
    responses = [
        '```json\n{"tool": "python_tool", "arguments": {"code": "print(21 * 2)"}}\n```',
        "The result of 21 * 2 is 42.",
    ]
    reasoner = MockReasoner(responses)
    agent = FoundationAgent(reasoner)
    agent.register_tool(PythonTool())

    events = []
    agent.subscribe(events.append)

    response = agent.run("Calculate 21 * 2")
    assert "42" in response.output
    assert response.iterations == 2
    assert response.tool_calls_count == 1

    # Verify events
    event_names = [e.event_name for e in events]
    assert "AgentStarted" in event_names
    assert "ToolStarted" in event_names
    assert "ToolFinished" in event_names
    assert "AgentFinished" in event_names


def test_agent_max_iterations_cap():
    # Infinite tool loop mock
    infinite_tool_response = (
        '```json\n{"tool": "python_tool", "arguments": {"code": "print(1)"}}\n```'
    )
    reasoner = MockReasoner(lambda p: infinite_tool_response)

    config = AgentConfig(max_iterations=3, verbose=False)
    agent = FoundationAgent(reasoner, config=config)
    agent.register_tool(PythonTool())

    response = agent.run("Loop test")
    assert response.iterations == 3
    assert response.tool_calls_count == 3
