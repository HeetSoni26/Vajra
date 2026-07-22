# Hugging Face Interoperability Guide

The `inference/hf_compat.py` adapter module provides round-trip compatibility between native `FoundationLM` checkpoints and Hugging Face `transformers` directories.

## Key Principles

- **Zero Core Modifications**: `FoundationLM` and `ModelConfig` remain the primary, unchanged core classes.
- **Thin Adapter Pattern**: Conversions translate architecture names and weight names cleanly.
- **Safetensors Support**: Automatically uses `safetensors` format with tied weight deduplication, falling back to `pytorch_model.bin` if safetensors is unavailable.

## Python API Usage

### Exporting to Hugging Face Format
```python
from model import FoundationLM, ModelConfig
from inference.hf_compat import save_pretrained

model_cfg = ModelConfig.from_yaml("configs/model/model_tiny.yaml")
model = FoundationLM(model_cfg)

save_pretrained(
    model=model,
    output_dir="checkpoints/hf_export",
    tokenizer_dir="tokenizer/v1.0"
)
```

### Loading from Hugging Face Directory
```python
from inference.hf_compat import load_pretrained

model, config = load_pretrained("checkpoints/hf_export", device="cpu")
```

### Round-Trip Checkpoint Conversion

```python
from inference.hf_compat import convert_checkpoint_to_hf, convert_hf_to_checkpoint

# Native .pt -> HF directory
convert_checkpoint_to_hf(
    checkpoint_path="checkpoints/run/latest.pt",
    model_config_path="configs/model/model_tiny.yaml",
    output_dir="checkpoints/hf_out",
    tokenizer_dir="tokenizer/v1.0"
)

# HF directory -> Native .pt
convert_hf_to_checkpoint(
    hf_dir="checkpoints/hf_out",
    output_path="checkpoints/restored.pt"
)
```
