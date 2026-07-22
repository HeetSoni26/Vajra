"""Multi-agent team building a feature example."""

from vajra_agent import MultiAgentEngine


def main():
    engine = MultiAgentEngine()
    engine.setup_default_team()

    response = engine.run("Implement JWT authentication with token refresh and unit tests.")
    print("Multi-Agent Team Response:")
    print(response.output)


if __name__ == "__main__":
    main()
