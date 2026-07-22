# Vajra-LM & Vajra-Agent Roadmap

This document outlines the version roadmap for the Vajra-LM engine and Vajra-Agent platform.

## [v1.0.0] - Production Release (Current)
- **Vajra-LM Engine**: Decoder-only Transformer, DDP, AMP, KV Cache, Hugging Face adapter, evaluation framework, FastAPI server, CLI.
- **Vajra-Agent Platform**:
  - Phase 1: Base Tooling, ToolRegistry, FunctionParser, Execution Loop, EventBus, MCP.
  - Phase 2: Task Planning Engine, Repository Intelligence, Workspace Indexing, PythonSandbox, VerificationEngine, ReflectionEngine.
  - Phase 3: MemoryManager, ContextBuilder, Embeddings Layer, Vector Storage, Semantic Retrieval, Knowledge Graph, Retention Policies.
  - Phase 4: MultiAgentEngine, Orchestrator, TaskGraph DAG engine, 10 Built-in Specialized Agents, PermissionManager, JobManager, ArtifactManager, AgentPluginRegistry, Stdio/Sse MCP Transports.
  - Phase 5: Observability Tracing, Performance Profiling, Preset Profiles, End-to-End Integration, Stress Testing, Showcase Applications.

## [v1.1.0] - Interactive UI & Concurrent Multi-Processing (Planned)
- Real-time Web & CLI Observability Dashboard (visualizing DAG task graph progress).
- Multi-process / thread-pool parallel worker dispatch for multi-agent readiness nodes.
- Direct OAuth2 / JWT authentication middleware for external MCP HTTP endpoints.

## [v2.0.0] - Self-Improving Engine & Autonomous CI/CD Pipelines (Planned)
- Closed-loop RLHF / DPO fine-tuning pipeline generating synthetic training data from agent execution trace logs.
- Native GitHub Actions & GitLab CI runner integrations for autonomous PR creation and review.
