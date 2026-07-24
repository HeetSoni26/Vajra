# Vajra Configuration System

[Overview](../README.md) | [Architecture](architecture.md) | [Training](training.md)

---

## Overview

Vajra uses clean, type-checked YAML configurations in `configs/` paired with Pydantic / dataclass schemas in Python to manage model hyper-parameters, training execution rules, and dataset mixtures.

---

## Configuration Hierarchy

```
configs/
├── model/                  # Model architecture specifications
│   ├── model_tiny.yaml     # Vajra-57M (8 layers, d_model=512)
│   ├── model_small.yaml    # Vajra-125M (12 layers, d_model=768)
│   └── model_base.yaml     # Vajra-370M (24 layers, d_model=1024)
├── training/               # Pretraining & fine-tuning parameters
│   ├── pretrain_tiny.yaml  # Fast pretraining profile
│   └── pretrain_full.yaml  # Full production pretraining profile
└── dataset/                # Dataset mixture and tokenization parameters
    ├── dataset_tiny.yaml   # Small validation mixture
    └── dataset_production.yaml # Production dataset sharding profile
```

---

## Key YAML Schemas

### 1. Model Configuration (`configs/model/model_tiny.yaml`)
```yaml
model_name: "vajra-lm-tiny"
vocab_size: 65536
hidden_size: 512
intermediate_size: 1376
num_layers: 8
num_attention_heads: 8
num_key_value_heads: 4
max_position_embeddings: 2048
rope_theta: 10000.0
dropout: 0.0
attention_dropout: 0.0
rmsnorm_eps: 1e-6
attention_bias: false
tie_word_embeddings: true
activation_function: "swiglu"
dtype: "bfloat16"
```

### 2. Programmatic Usage in Python
```python
from utils.config import load_config
from model.architecture import ModelConfig

# Load YAML configuration into structured dataclass
raw_cfg = load_config("configs/model/model_tiny.yaml")
model_cfg = ModelConfig(**raw_cfg)

print(f"Loaded config for {model_cfg.model_name}: {model_cfg.hidden_size} hidden size")
```
