"""Documentation generation workflow example."""

from vajra_agent import DocGenerationWorkflow


def main():
    wf = DocGenerationWorkflow()
    response = wf.execute("vajra_agent/multi_agent")

    print("Documentation Generation Output:")
    print(response.output)


if __name__ == "__main__":
    main()
