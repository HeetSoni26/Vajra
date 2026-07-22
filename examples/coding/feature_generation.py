"""Feature generation workflow example."""

from vajra_agent import FoundationAgent, MockReasoner
from vajra_agent.workflows import FeatureWorkflow


def main():
    responses = [
        "Implemented JWT authentication endpoints in api/routes/auth.py and generated test cases."
    ]
    reasoner = MockReasoner(responses)
    agent = FoundationAgent(reasoner)

    wf = FeatureWorkflow(agent)
    res = wf.execute("Implement JWT authentication with token refresh")

    print("Feature Workflow Output:")
    print(res.output)


if __name__ == "__main__":
    main()
