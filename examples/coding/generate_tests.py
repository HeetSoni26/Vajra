"""Test generation workflow example."""

from vajra_agent import FoundationAgent, MockReasoner
from vajra_agent.workflows import TestGenerationWorkflow


def main():
    responses = ["Generated pytest unit test suite covering 100% of methods in model/config.py."]
    reasoner = MockReasoner(responses)
    agent = FoundationAgent(reasoner)

    wf = TestGenerationWorkflow(agent)
    res = wf.execute(source_file="model/config.py", test_file="tests/test_config_gen.py")

    print("Test Generation Output:")
    print(res.output)


if __name__ == "__main__":
    main()
