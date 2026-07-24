# Section 10 — Deployment Plan

## 10.1 Objective

Release the model in formats usable by researchers, local-inference users, and service deployments. The deployment plan covers HuggingFace Hub, GGUF/llama.cpp, Ollama, vLLM, FastAPI, Docker, Python SDK, and Gradio.

Implementation entry points:

- `deployment/upload_hf.py`
- `deployment/Modelfile`
- `deployment/gradio_app.py`
- `Dockerfile.serve`
- `requirements-serve.txt`
- `api/`
- `sdk/`
- `inference/convert/`

## 10.2 HuggingFace Hub release

Required files in the final HuggingFace model repository:

```text
config.json
generation_config.json
tokenizer.json
tokenizer_config.json
special_tokens_map.json
model.safetensors or model shards
README.md model card
```

Upload flow:

```bash
python deployment/upload_hf.py \
  --folder checkpoints/final/hf \
  --repo_id yourname/vajra-lm-1b
```

Validation:

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

model = AutoModelForCausalLM.from_pretrained("yourname/vajra-lm-1b")
tok = AutoTokenizer.from_pretrained("yourname/vajra-lm-1b")
```

## 10.3 GGUF and llama.cpp

Convert and quantize:

```bash
python inference/convert/to_gguf.py \
  --llama_cpp /path/to/llama.cpp \
  --model checkpoints/final/hf \
  --outfile models/gguf/model-f16.gguf

python inference/convert/quantize_gguf.py \
  --quantizer /path/to/llama-quantize \
  --input models/gguf/model-f16.gguf \
  --output models/gguf/model-Q4_K_M.gguf \
  --type Q4_K_M
```

Publish at least:

- Q4_K_M for CPU-first local inference
- Q5_K_M for better quality/size tradeoff
- Q8_0 for near-lossless local inference

## 10.4 Ollama

Use `deployment/Modelfile` after generating a GGUF checkpoint.

```bash
ollama create vajra-lm -f deployment/Modelfile
ollama run vajra-lm
```

Validation: the model should accept normal prompts, stop on configured special tokens, and use the intended chat template.

## 10.5 vLLM serving

Recommended for high-throughput GPU serving:

```python
from vllm import LLM, SamplingParams

llm = LLM(
    model="yourname/vajra-lm-1b-instruct",
    dtype="bfloat16",
    gpu_memory_utilization=0.85,
    max_model_len=4096,
)
params = SamplingParams(temperature=0.7, max_tokens=256)
outputs = llm.generate(["Explain grouped-query attention."], params)
```

Record throughput and latency for batch sizes 1, 4, 8, and 16.

## 10.6 FastAPI REST API

The scaffold includes OpenAI-style routes:

- `GET /health`
- `GET /v1/models`
- `POST /v1/completions`
- `POST /v1/chat/completions`

Run locally after installing serving dependencies:

```bash
uvicorn api.main:app --host 0.0.0.0 --port 8000
```

The checked-in route implementations are safe stubs. Replace stub generation with model-backed inference after final checkpoint export.

## 10.7 Docker serving container

Build and run:

```bash
docker build -t vajra-lm:serve -f Dockerfile.serve .
docker run --gpus all -p 8000:8000 vajra-lm:serve
```

Validation:

```bash
curl http://localhost:8000/health
curl http://localhost:8000/v1/models
```

## 10.8 Python SDK

The SDK package lives in `sdk/`.

Example usage:

```python
from foundationlm import FoundationLMClient

client = FoundationLMClient(base_url="http://localhost:8000")
print(client.complete("The capital of France is"))
print(client.chat([{"role": "user", "content": "Explain RoPE."}]))
```

## 10.9 Gradio demo

The demo is in `deployment/gradio_app.py`. It should use the instruct checkpoint, not the base checkpoint. The demo should state model limitations and avoid implying benchmark results that have not been measured.

## 10.10 Release validation checklist

- HuggingFace load works through `AutoModelForCausalLM` and `AutoTokenizer`.
- GGUF models load in llama.cpp.
- Ollama model runs with the correct template.
- FastAPI container starts and responds to health checks.
- SDK can call completion and chat endpoints.
- Gradio demo loads and produces text.
- Model card and README are complete.
- License and dataset documentation are reviewed.
