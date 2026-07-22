# Vajra-Agent Phase 4 — Autonomous Agent Platform & Multi-Agent Orchestration Documentation

Phase 4 transforms `Vajra-Agent` into a complete autonomous agent platform capable of orchestrating multi-agent teams, managing DAG execution task graphs, running background jobs, executing re-usable workflows, and evaluating human approval policies.

## Multi-Agent Architecture

```
                          ┌─────────────────────────────┐
                          │      MultiAgentEngine       │
                          └──────────────┬──────────────┘
                                         │
                                         ▼
                          ┌─────────────────────────────┐
                          │         Orchestrator        │
                          └──────────────┬──────────────┘
                                         │
    ┌──────────────────┬─────────────────┼──────────────────┬──────────────────┐
    ▼                  ▼                 ▼                  ▼                  ▼
┌──────────────┐ ┌──────────────┐ ┌──────────────┐  ┌──────────────┐  ┌──────────────────┐
│ TaskGraph    │ │ SharedMemory │ │ JobManager   │  │ Permission   │  │ ArtifactManager  │
│ Execution DAG│ │ & Vector DB  │ │ & Background │  │ Manager      │  │ Versioned Store  │
└──────────────┘ └──────────────┘ └──────────────┘  └──────────────┘  └──────────────────┘
```

## Core Components

### 1. Specialized Agents (`vajra_agent/specialized/`)
- 10 Built-in specialized agent roles: `ArchitectAgent`, `PlannerAgent`, `ResearchAgent`, `CoderAgent`, `ReviewerAgent`, `TesterAgent`, `DebuggerAgent`, `DocumentationAgent`, `SecurityAgent`, `RefactorAgent`.

### 2. Orchestrator & Task Graph (`vajra_agent/multi_agent/`)
- `TaskGraph`: DAG topology resolving task dependencies, parallel tasks, retries, and failure propagation.
- `Orchestrator`: Assigns task nodes to specialized agents, dispatches structured messages, and synthesizes outputs.

### 3. Permission Approval Manager (`vajra_agent/approval/`)
- `PermissionManager`: Evaluates action categories (`FILE_DELETE`, `SHELL_EXECUTION`, `GIT_PUSH`, etc.) against policies (`ALWAYS_ALLOW`, `ALWAYS_DENY`, `ASK_EVERY_TIME`).

### 4. Job & Artifact Management (`vajra_agent/jobs/` & `artifacts/`)
- `JobManager`: Manages background task state (`PENDING`, `RUNNING`, `PAUSED`, `COMPLETED`, `FAILED`, `CANCELLED`).
- `ArtifactManager`: Stores versioned execution artifacts, diffs, plans, logs, and verification reports.

### 5. Reusable Multi-Agent Workflows (`vajra_agent/multi_agent/workflows/`)
- `SaaSBuildWorkflow`, `RepoRefactorWorkflow`, `FixFailingTestsWorkflow`, `DocGenerationWorkflow`, `SecurityAuditWorkflow`.
