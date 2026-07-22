"""Conversation memory example with FoundationAgent."""

from vajra_agent import FoundationAgent, MockReasoner
from vajra_agent.memory import MemoryManager


def main():
    mem = MemoryManager()
    mem.remember("User preferred coding style: Strict type annotations with Python 3.11 features.")

    reasoner = MockReasoner(["I will generate Python 3.11 code with strict type annotations."])
    agent = FoundationAgent(reasoner)
    agent.attach_memory(mem)

    response = agent.run("Generate a helper function for string formatting.")
    print("Agent Output with Memory:")
    print(response.output)


if __name__ == "__main__":
    main()
