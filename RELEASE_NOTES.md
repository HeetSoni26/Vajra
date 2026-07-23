# Vajra v1.0.0 Release Notes

**Release Date**: July 23, 2026  
**Tag**: `v1.0.0`  
**License**: Apache-2.0  
**Repository**: [https://github.com/HeetSoni26/Vajra](https://github.com/HeetSoni26/Vajra)

---

## 🚀 Summary

We are excited to announce the official **v1.0.0 Open Source Release** of the **Vajra Framework**!

Vajra is an end-to-end open-source AI ecosystem combining a high-performance decoder-only language model architecture (**Vajra-LM**) with an advanced autonomous agent framework (**Vajra-Agent**).

> [!NOTE]  
> **Framework Status Statement**:  
> The Vajra Framework codebase, multi-agent engine, inference APIs, and training infrastructure are production-ready with **100% test pass rate (180/180 tests)** and verified GitHub Actions CI workflows.  
> Model weights pre-training for the Vajra-370M and Vajra-1B models will follow in a separate dedicated release.

---

## 🔥 Key Highlights

### 1. Vajra-LM Core Architecture
- **Decoder-Only Transformer**: LLaMA-style architecture featuring SwiGLU activation, RoPE positional embeddings, RMSNorm, and Grouped-Query Attention (GQA).
- **HuggingFace Compatibility**: Native `AutoConfig` and `AutoModelForCausalLM` integration.
- **Inference Server**: FastAPI server with asynchronous Server-Sent Events (SSE) streaming for `/generate`, `/v1/completions`, and `/v1/chat/completions`.
- **Training Engine**: Distributed Data Parallel (DDP) engine with hardware-agnostic AMP (`GradScaler`), SFT, and DPO alignment routines.

### 2. Vajra-Agent Framework
- **Multi-Agent Orchestration**: `MultiAgentEngine`, `Orchestrator`, DAG `TaskGraph` engine, and `SharedMemory`.
- **10 Specialized Built-in Agents**: Architect, Coder, Tester, Reviewer, SecurityAuditor, DevOps, DataEngineer, Researcher, DocumentationSpecialist, and ProjectManager.
- **Coding Intelligence**: Sandboxed Python execution (`PythonSandbox`), shell execution (`ShellTool`), indexing, context building, and verification pipelines.
- **Memory Subsystem**: Vector storage, knowledge graph, semantic retrieval, and memory retention policies.

### 3. Production & Security Engineering
- **Docker Hardened**: Non-root execution (`USER vajra`), healthchecks, and production `docker-compose.yml`.
- **Type Safety**: PEP 561 `py.typed` markers included across all package modules.
- **Security Policy**: Integrated with [GitHub Security Advisories](https://github.com/HeetSoni26/Vajra/security/advisories).

---

## 📦 Installation

```bash
# Clone repository
git clone https://github.com/HeetSoni26/Vajra.git
cd Vajra

# Install full environment
pip install -e .[all]
```

## 📄 License & Attribution

Licensed under the **Apache License 2.0**.
