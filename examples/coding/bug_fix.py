"""Bug fix workflow example."""

from vajra_agent import FoundationAgent, MockReasoner
from vajra_agent.workflows import BugFixWorkflow


def main():
    responses = [
        "Diagnosed bug in calc.py. Fixed division by zero check and verified with pytest."
    ]
    reasoner = MockReasoner(responses)
    agent = FoundationAgent(reasoner)

    wf = BugFixWorkflow(agent)
    res = wf.execute(bug_description="ZeroDivisionError when dividing by 0 in calc.py", file_target="calc.py")

    print("Bug Fix Workflow Output:")
    print(res.output)


if __name__ == "__main__":
    main()
