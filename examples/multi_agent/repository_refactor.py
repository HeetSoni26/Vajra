"""Repository refactor workflow example."""

from vajra_agent import RepoRefactorWorkflow


def main():
    wf = RepoRefactorWorkflow()
    response = wf.execute("vajra_agent/memory", "Extract base storage interfaces")

    print("Refactor Workflow Output:")
    print(response.output)


if __name__ == "__main__":
    main()
