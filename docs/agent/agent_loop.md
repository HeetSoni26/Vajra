# Agent Execution Loop

The core execution loop (`vajra_agent/agent/agent.py`) processes user tasks iteratively.

## Loop Lifecycle Diagram

```
       ┌───────────────────────────────┐
  ┌───>│            OBSERVE            │
  │    │  Inspect conversation state   │
  │    └───────────────┬───────────────┘
  │                    │
  │                    ▼
  │    ┌───────────────────────────────┐
  │    │             THINK             │
  │    │  Query Reasoner with schemas  │
  │    └───────────────┬───────────────┘
  │                    │
  │                    ▼
  │    ┌───────────────────────────────┐
  │    │             PLAN              │
  │    │  Multi-step planning hook     │
  │    └───────────────┬───────────────┘
  │                    │
  │                    ▼
  │    ┌───────────────────────────────┐
  │    │              ACT              │
  │    │ Parse JSON -> Execute Tool    │
  │    └───────────────┬───────────────┘
  │                    │
  │     [Has Tool Call?]
  ├─── YES ────────────┴───── NO ────────┐
  │                                      │
  │    ┌───────────────────────────────┐ │
  │    │            REFLECT            │ │
  │    │ Record result & emit event    │ │
  │    └───────────────────────────────┘ │
  │                                      ▼
  └─────────────────────────────── Final Answer
```

## Features

- **Recursion Cap**: Controlled via `config.max_iterations` (default: 10).
- **Error Gracefulness**: Captures tool errors in `ToolExecutionResult` and injects error messages back into the conversation for self-correction.
- **Event Telemetry**: Emits lifecycle events (`AgentStarted`, `ToolStarted`, `ToolFinished`, `AgentFinished`) via `EventBus`.
