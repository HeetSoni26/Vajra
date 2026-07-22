# Vajra-Agent Architecture Overview

`Vajra-Agent` is an autonomous AI Software Engineer reasoning and tool execution framework built on top of `Vajra-LM`.

## System Overview

```
                        ┌───────────────────────────────┐
                        │          User Query           │
                        └───────────────┬───────────────┘
                                        │
                                        ▼
                        ┌───────────────────────────────┐
                        │        FoundationAgent        │
                        └───────────────┬───────────────┘
                                        │
             ┌──────────────────────────┼──────────────────────────┐
             ▼                          ▼                          ▼
┌─────────────────────────┐ ┌──────────────────────┐ ┌───────────────────┐
│     BaseReasoner        │ │     ToolRegistry     │ │    EventBus       │
│  (FoundationReasoner /  │ │ (File, Shell, Python │ │  (Observability / │
│    MockReasoner)        │ │   Git, MCP Adapters) │ │   Telemetry)      │
└─────────────────────────┘ └──────────────────────┘ └───────────────────┘
```

## Key Architectural Principles

1. **Frozen Vajra Model**: `Vajra-LM` is frozen and completely independent. `Vajra-Agent` imports and wraps it cleanly.
2. **Reasoner Interface**: `FoundationAgent` depends only on `BaseReasoner`, allowing seamless swapping between native `FoundationLM`, Hugging Face, Ollama, OpenAI APIs, or mock test engines.
3. **Observe-Think-Plan-Act-Reflect Loop**: Structured internal execution lifecycle designed for future multi-step planning and self-reflection extensions.
4. **Strongly Typed Tool Results**: All tools return structured `ToolExecutionResult` models (`success`, `output`, `error`, `execution_time_ms`).
5. **Decoupled Tool System**: Tools are standalone python objects exposing JSON schemas and execution handlers without knowing about the LLM.
