"""Basic Vajra-Agent example."""

from vajra_agent import FoundationAgent, MockReasoner


def main():
    # Use MockReasoner for demonstration without a GPU
    reasoner = MockReasoner(
        [
            "To build an AI software engineer, you need reasoning, tool execution, and an execution loop."
        ]
    )

    agent = FoundationAgent(reasoner)
    response = agent.run("What are the key pillars of an AI software engineer?")

    print("Agent Final Answer:")
    print(response.output)
    print(f"\nCompleted in {response.iterations} iteration(s).")


if __name__ == "__main__":
    main()
