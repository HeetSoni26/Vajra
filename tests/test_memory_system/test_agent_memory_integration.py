"""Integration tests verifying FoundationAgent attached with MemoryManager."""

from vajra_agent import FoundationAgent, MockReasoner
from vajra_agent.memory import MemoryManager


def test_agent_memory_integration():
    mem = MemoryManager()
    mem.remember("Architectural decision: Use JWT tokens for API auth.")

    reasoner = MockReasoner(["To authenticate users, use JWT tokens as decided."])
    agent = FoundationAgent(reasoner)
    agent.attach_memory(mem)

    response = agent.run("How should we handle API authentication?")
    assert "JWT" in response.output
    assert response.iterations == 1
