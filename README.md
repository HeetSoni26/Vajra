# Vajra

<div align="center">
  <img src="https://img.shields.io/badge/Vajra-v1.0.0-blue.svg" alt="Version">
  <img src="https://img.shields.io/badge/License-Apache%202.0-green.svg" alt="License">
  <img src="https://img.shields.io/badge/Python-3.10%2B-blue.svg" alt="Python">
</div>

**Vajra** (formerly Foundation) is a production-ready, open-source AI ecosystem combining a high-performance foundation language model (**Vajra-LM**) with an advanced autonomous agent framework (**Vajra-Agent**).

## 🌟 Vision

Vajra provides an end-to-end framework where the model (Vajra-LM) "thinks" and the agent (Vajra-Agent) "acts". It is built for researchers, developers, and enterprises who need full control over both their model architecture and multi-agent orchestration layer.

## 🚀 Major Features

### Vajra-LM
*   **Decoder-Only Transformer**: Highly optimized LLaMA-style architecture.
*   **Advanced Training**: Native support for DDP, AMP, and Gradient Checkpointing.
*   **Production Inference**: KV Caching and high-throughput generation.
*   **Hugging Face Compatibility**: Seamlessly use Vajra models with HF pipelines.
*   **API Ready**: Built-in FastAPI integration for production deployment.

### Vajra-Agent
*   **Coding Intelligence**: Task Planning, Repository Scanning, Python Sandboxing, and Automated Verification.
*   **Semantic Memory**: Persistent memory subsystem, semantic retrieval, and repository knowledge graphs.
*   **Multi-Agent Orchestration**: DAG task graph engine, `MultiAgentEngine`, and 10 specialized built-in agents (Architect, Coder, Tester, Reviewer, etc.).
*   **Production Readiness**: Observability tracing, customizable agent profiles, and automated benchmarking.

## 📁 Repository Structure

```text
vajra/
├── vajra_agent/        # The Vajra Autonomous Agent Framework
├── model/              # Vajra-LM core architecture
├── api/                # FastAPI serving endpoints
├── cli/                # Command-line interface tools
├── docs/               # Comprehensive documentation
├── examples/           # End-to-end multi-agent and training examples
├── tests/              # Extensive unit and integration test suite
└── validation/         # Real-world validation suite and reports
```

## 🛠️ Installation

```bash
# Clone the repository
git clone https://github.com/vajra-ai/vajra.git
cd vajra

# Install with all dependencies
pip install -e .[all]
```

## ⚡ Quick Start

### Vajra-Agent Multi-Agent Workflow
```python
from vajra_agent import FoundationAgent, MultiAgentEngine
from vajra_agent.reasoners import VajraReasoner

# Initialize the reasoner with Vajra-LM
reasoner = VajraReasoner(model_path="path/to/vajra-checkpoint")

# Run a single agent task
agent = FoundationAgent(reasoner=reasoner)
result = agent.run("Refactor the model loading logic in utils.py")
print(result.output)
```

### Vajra-LM Inference
```python
from model import VajraForCausalLM, VajraConfig

config = VajraConfig.from_pretrained("path/to/checkpoint")
model = VajraForCausalLM.from_pretrained("path/to/checkpoint")
```

## 📊 Benchmarks & Validation

Vajra v1.0.0 has passed a rigorous **Real-World Validation Suite** covering multi-language repository scanning, 10 distinct software engineering tasks, and end-to-end multi-agent orchestration scenarios.

*   **Task Success Rate**: 100.0%
*   **Verification Success**: 100.0%
*   **Pytest Suite**: 180/180 Passed
*   **Ruff Cleanliness**: 0 Errors

See the [Validation Report](docs/agent/validation_report.md) for full metrics.

## 🗺️ Roadmap
See [ROADMAP.md](ROADMAP.md) for future plans including larger models (Vajra-370M, Vajra-1B).

## 🤝 Contributing
We welcome contributions! Please see our [Contributing Guide](CONTRIBUTING.md) and [Code of Conduct](CODE_OF_CONDUCT.md).

## 📄 License
This project is licensed under the **Apache License 2.0**. See the [LICENSE](LICENSE) file for details.
