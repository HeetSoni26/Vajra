"""Full Production Suite Showcase demonstrating end-to-end integration of all Vajra-Agent subsystems."""

from vajra_agent import (
    ActionCategory,
    AgentPluginRegistry,
    ArtifactManager,
    CoderAgent,
    JobManager,
    MemoryManager,
    MultiAgentEngine,
    PermissionManager,
    PermissionPolicy,
    SaaSBuildWorkflow,
    StdioTransport,
)
from vajra_agent.observability import ExecutionTrace


def main():
    print("==================================================")
    print(" Vajra-Agent v1.0.0 Full Production Suite   ")
    print("==================================================")

    # 1. Execution Tracing & Observability
    tracer = ExecutionTrace("trace_prod_001")
    span1 = tracer.start_span("init_system")

    # 2. Permission Policy Manager
    pm = PermissionManager(default_policy=PermissionPolicy.ALWAYS_ALLOW)
    pm.set_policy(ActionCategory.FILE_DELETE, PermissionPolicy.ALWAYS_DENY)

    # 3. Artifact & Job Managers
    am = ArtifactManager()
    jm = JobManager()

    job = jm.submit_job("Background indexing & compilation", fn=lambda: "Indexed 100% of workspace")
    am.create_artifact(
        kind="plan", name="saas_master_plan", content="Phase 1: Auth, Phase 2: Billing"
    )

    # 4. MemoryManager Repository Indexing
    memory = MemoryManager()
    memory.index_repository(".")

    tracer.end_span(span1)

    # 5. MCP Transport Setup
    mcp_stdio = StdioTransport("python", ["--version"])
    mcp_stdio.connect()

    # 6. Marketplace Plugin Registration
    AgentPluginRegistry.register_agent_plugin("CustomCoderRole", CoderAgent)

    # 7. Multi-Agent Engine Workflow
    span2 = tracer.start_span("execute_multi_agent_workflow")

    engine = MultiAgentEngine(shared_memory=None)
    engine.setup_default_team()

    workflow = SaaSBuildWorkflow(engine=engine)
    response = workflow.execute(
        feature_description="Build JWT authentication and Stripe payments billing system"
    )

    tracer.end_span(span2)
    tracer.finish()
    tracer.save("checkpoints/traces/execution_trace.json")

    print("\n--- System Execution Report ---")
    print(f"Workflow Iterations: {response.iterations}")
    print(f"Job Status: {job.status}")
    print(
        f"Latest Artifact: {am.get_latest_artifact('saas_master_plan').name} v{am.get_latest_artifact('saas_master_plan').version}"
    )
    print(f"Indexed Repository Framework: {memory.project_context.repo_context.framework}")
    print(f"Trace Duration: {tracer.to_dict()['duration_s']}s")
    print("\nFinal Output Summary:")
    print(response.output)

    mcp_stdio.close()


if __name__ == "__main__":
    main()
