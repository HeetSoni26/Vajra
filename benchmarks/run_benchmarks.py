"""Performance profiling and benchmarking suite measuring Vajra-Agent subsystem latencies."""

from __future__ import annotations

import json
from pathlib import Path
import time

from vajra_agent import (
    ContextBuilder,
    FoundationAgent,
    MemoryManager,
    MockReasoner,
    MultiAgentEngine,
    PlanningEngine,
    ReflectionEngine,
    RepositoryScanner,
    VerificationEngine,
    WorkspaceIndexer,
)
from vajra_agent.schemas.state import AgentState


def run_all_benchmarks() -> dict[str, float]:
    results: dict[str, float] = {}

    # 1. Planning Engine Latency
    reasoner = MockReasoner(["Plan step 1, step 2"])
    planner = PlanningEngine(reasoner)
    t0 = time.perf_counter()
    planner.generate_plan("Build API", [])
    results["planning_ms"] = round((time.perf_counter() - t0) * 1000, 2)

    # 2. Repository Scanning & Workspace Indexing
    t0 = time.perf_counter()
    RepositoryScanner.scan(".")
    results["repo_scanning_ms"] = round((time.perf_counter() - t0) * 1000, 2)

    t0 = time.perf_counter()
    WorkspaceIndexer.index_directory(".")
    results["workspace_indexing_ms"] = round((time.perf_counter() - t0) * 1000, 2)

    # 3. Memory Retrieval & MemoryManager Indexing
    mem = MemoryManager()
    t0 = time.perf_counter()
    mem.index_repository(".")
    results["memory_repo_indexing_ms"] = round((time.perf_counter() - t0) * 1000, 2)

    t0 = time.perf_counter()
    mem.recall("FoundationAgent", top_k=5)
    results["memory_recall_ms"] = round((time.perf_counter() - t0) * 1000, 2)

    # 4. Context Construction
    state = AgentState()
    state.conversation.add_user("User task query")
    t0 = time.perf_counter()
    ContextBuilder.build_enriched_context(
        state=state,
        tool_schemas=[],
        project_context=mem.project_context,
        knowledge_graph=mem.knowledge_graph,
    )
    results["context_construction_ms"] = round((time.perf_counter() - t0) * 1000, 2)

    # 5. Verification Engine
    t0 = time.perf_counter()
    VerificationEngine.verify(run_linter=False, run_tests=False)
    results["verification_engine_ms"] = round((time.perf_counter() - t0) * 1000, 2)

    # 6. Reflection Engine
    reflector = ReflectionEngine()
    t0 = time.perf_counter()
    reflector.reflect(state)
    results["reflection_engine_ms"] = round((time.perf_counter() - t0) * 1000, 2)

    # 7. Single Agent Task Run
    agent = FoundationAgent(reasoner)
    t0 = time.perf_counter()
    agent.run("Execute query task")
    results["single_agent_execution_ms"] = round((time.perf_counter() - t0) * 1000, 2)

    # 8. Multi-Agent Orchestration
    engine = MultiAgentEngine()
    engine.setup_default_team()
    t0 = time.perf_counter()
    engine.run("Build SaaS Feature")
    results["multi_agent_orchestration_ms"] = round((time.perf_counter() - t0) * 1000, 2)

    # Save benchmark_report.json
    out_dir = Path("benchmarks")
    out_dir.mkdir(parents=True, exist_ok=True)
    report_file = out_dir / "benchmark_report.json"
    report_file.write_text(json.dumps(results, indent=2), encoding="utf-8")

    return results


if __name__ == "__main__":
    benchmarks = run_all_benchmarks()
    print("Vajra-Agent Subsystem Performance Benchmarks (ms):")
    for k, v in benchmarks.items():
        print(f"  - {k}: {v} ms")
