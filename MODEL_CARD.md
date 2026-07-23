# Vajra-LM Model Card

## Model Details

- **Model Name**: Vajra-LM (Vajra-370M Framework Target)
- **Developer**: Vajra Open Source Team
- **Model Type**: Decoder-Only Transformer Language Model
- **License**: Apache 2.0
- **Framework Version**: v1.0.0
- **Status**: **Framework Complete — Model Weights Pending Real Pre-training Run**

> [!IMPORTANT]  
> **Training Disclaimer**: The Vajra codebase and execution framework are production-ready (100% test pass rate, verified inference, distributed DDP training engine). Model weights presented here are initial architecture structural checkpoints or randomly initialized weights for testing and verification. Full pre-trained weights for the 370M and 1B parameter variants will be published in a future model weights release.

## Architecture

- **Attention Mechanism**: Grouped-Query Attention (GQA) with Rotary Position Embeddings (RoPE)
- **Normalization**: RMSNorm with configurable epsilon (`1e-5`)
- **Activation**: SwiGLU non-linear activation
- **Vocabulary Size**: 32,000 (Byte-Pair Encoding with HuggingFace Tokenizers compatibility)

## Quick Start (Inference)

```python
from model import VajraForCausalLM, VajraConfig
from tokenizer import VajraTokenizer

# Load architecture configuration
config = VajraConfig(
    vocab_size=32000,
    hidden_size=1024,
    intermediate_size=2816,
    num_layers=24,
    num_attention_heads=16,
    num_key_value_heads=8,
)

# Instantiate model
model = VajraForCausalLM(config)
model.eval()

# Tokenize prompt
tokenizer = VajraTokenizer.from_pretrained("vajra-ai/vajra-tokenizer")
inputs = tokenizer("Definitive guide to foundation model architectures:", return_tensors="pt")

# Generate response
outputs = model.generate(**inputs, max_new_tokens=50)
print(tokenizer.decode(outputs[0], skip_special_tokens=True))
```

## Intended Use

- **Primary Uses**: Research in efficient language modeling, multi-agent orchestration via Vajra-Agent, and customized domain fine-tuning (SFT/DPO).
- **Out of Scope**: High-risk autonomous decision making without human review prior to full pre-training completion.

## Citation

```bibtex
@software{vajra2026,
  author = {Vajra AI Team},
  title = {Vajra: Open-Source Foundation Model & Autonomous Multi-Agent Ecosystem},
  year = {2026},
  version = {1.0.0},
  url = {https://github.com/HeetSoni26/Vajra}
}
```
