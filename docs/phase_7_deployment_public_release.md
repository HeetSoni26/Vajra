# Phase 7 — Deployment & Public Release

## Objective

Package and publish the model, code, documentation, and technical report as a complete open-source release.

## Deliverables included

- `deployment/upload_hf.py`
- `deployment/Modelfile`
- `deployment/gradio_app.py`
- `Dockerfile.serve`
- `requirements-serve.txt`
- `api/` OpenAI-compatible route skeletons
- `sdk/` Python client package

## Release checklist

- HuggingFace Hub repository contains model weights, tokenizer files, generation config, and model card.
- GGUF files are uploaded for llama.cpp and Ollama users.
- Docker server image builds and exposes `/v1/completions`, `/v1/chat/completions`, `/v1/models`, and `/health`.
- Python SDK can call the REST API.
- Technical report and model card document training data, intended uses, limitations, and benchmark results.

## Validation criteria

- `AutoModelForCausalLM.from_pretrained(repo_id)` works.
- `ollama run vajra-lm` works after creating the Ollama model from the Modelfile.
- Docker container starts and returns a valid health response.
- HuggingFace Space demo runs with the instruct checkpoint.
