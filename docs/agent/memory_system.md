# Vajra-Agent Phase 3 — Memory & Knowledge System Documentation

Phase 3 introduces persistent memory, vector storage, semantic retrieval, repository knowledge graphs, context builders, retention policies, and automatic conversation summarization for `Vajra-Agent`.

## Memory Subsystem Overview

```
                          ┌─────────────────────────────┐
                          │        MemoryManager        │
                          └──────────────┬──────────────┘
                                         │
    ┌──────────────────┬─────────────────┼─────────────────┬──────────────────┐
    ▼                  ▼                 ▼                 ▼                  ▼
┌──────────────┐ ┌──────────────┐ ┌──────────────┐ ┌──────────────┐ ┌──────────────────┐
│ Short-Term   │ │ Long-Term    │ │ Embeddings   │ │ Vector Store │ │ Knowledge Graph  │
│ Working Mem  │ │ Memory Store │ │ Provider     │ │ (InMemory /  │ │ (Symbol Node /   │
│ & Summaries  │ │ (LocalDisk)  │ │ (Mock / HF)  │ │  LocalDisk)  │ │  Edge Topology)  │
└──────────────┘ └──────────────┘ └──────────────┘ └──────────────┘ └──────────────────┘
```

## Core Components

### 1. MemoryManager (`vajra_agent/memory/manager.py`)
Central coordinator interface used by `FoundationAgent`:
- `remember(text, metadata)`: Embeds and stores text in vector memory.
- `recall(query, top_k)`: Queries semantic memory using similarity and recency weighting.
- `index_repository(path)`: Ingests codebase into `RepositoryKnowledgeGraph` and vector store.

### 2. Embedding Engine (`vajra_agent/memory/embeddings/`)
- `BaseEmbeddingProvider`: Abstract provider interface.
- `MockEmbeddingProvider`: Deterministic hash-based unit-normalized float vector generator.

### 3. Vector Storage (`vajra_agent/memory/storage/`)
- `BaseVectorStore`: Abstract vector store contract.
- `InMemoryVectorStore`: High-performance cosine similarity in-memory search.
- `LocalDiskVectorStore`: Disk-backed vector store persisting state to JSON.

### 4. Semantic Retrieval Engine (`vajra_agent/memory/retrieval/`)
`RetrievalEngine` calculates combined score: $\text{Score} = 0.8 \cdot \text{CosineSim} + 0.2 \cdot \text{RecencyWeight}$.

### 5. Repository Knowledge Graph (`vajra_agent/memory/knowledge/`)
`RepositoryKnowledgeGraph` maps architectural relationships between projects, files, classes, functions, and imports.

### 6. Memory Summarization & Retention Policies (`vajra_agent/memory/summaries/` & `policies/`)
`MemorySummarizer` compresses conversations when exceeding context thresholds. `LRUPolicy`, `ImportancePolicy`, and `RecentOnlyPolicy` handle memory eviction.

### 7. Context Builder (`vajra_agent/memory/context_builder.py`)
`ContextBuilder` constructs full prompt context combining system prompts, tool schemas, project context, knowledge graphs, retrieved memories, active plans, and reflection notes.
