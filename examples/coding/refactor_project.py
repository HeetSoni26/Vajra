"""Refactor workflow example."""

from vajra_agent import FoundationAgent, MockReasoner
from vajra_agent.workflows import RefactorWorkflow


def main():
    responses = [
        "Refactored model.py by extracting feedforward SwiGLU logic into a dedicated module."
    ]
    reasoner = MockReasoner(responses)
    agent = FoundationAgent(reasoner)

    wf = RefactorWorkflow(agent)
    res = wf.execute(refactor_goal="Extract SwiGLU block", target_file="model/model.py")

    print("Refactor Workflow Output:")
    print(res.output)


if __name__ == "__main__":
    main()
