"""Project Context and Workspace Index summary example."""

from vajra_agent import ProjectContext, WorkspaceIndexer


def main():
    ctx = ProjectContext.load(".")
    index = WorkspaceIndexer.index_directory(".")

    print("Project Context Summary:")
    print(f"  Root: {ctx.workspace_root}")
    print(f"  Language: {ctx.repo_context.primary_language}")
    print(f"  Framework: {ctx.repo_context.framework}")

    print("\nWorkspace Index Summary:")
    print(f"  Indexed Files: {len(index.files)}")
    print(f"  Classes Found: {len(index.get_classes())}")
    print(f"  Functions Found: {len(index.get_functions())}")


if __name__ == "__main__":
    main()
