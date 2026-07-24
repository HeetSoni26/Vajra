"""Repository analysis workflow example."""

from vajra_agent import FoundationAgent, MockReasoner
from vajra_agent.repository import RepositoryScanner
from vajra_agent.workflows import RepoAnalysisWorkflow


def main():
    ctx = RepositoryScanner.scan(".")
    print("Scanned Repository Context:")
    print(f"  Project Root: {ctx.project_root}")
    print(f"  Primary Language: {ctx.primary_language}")
    print(f"  Package Manager: {ctx.package_manager}")
    print(f"  Framework: {ctx.framework}")
    print(f"  Total Files: {ctx.total_files}")

    reasoner = MockReasoner(
        ["Repository analysis completed. Project is a Python FastAPI codebase."]
    )
    agent = FoundationAgent(reasoner)

    wf = RepoAnalysisWorkflow(agent)
    res = wf.execute(".")
    print("\nWorkflow Analysis Output:")
    print(res.output)


if __name__ == "__main__":
    main()
