# Vajra LM Command-Line Interface (CLI)

The `vajra-lm` CLI provides unified access to text generation, chat, evaluation, tokenization, and performance profiling.

## Installation

```bash
pip install -e .
```

## Commands Reference

### 1. `vajra-lm generate`
Generate text from a prompt.
```bash
vajra-lm generate \
  --prompt "In a world driven by AI" \
  --config configs/training/pretrain_tiny.yaml \
  --max_tokens 64 \
  --temperature 0.7 \
  --top_k 50 \
  --top_p 0.9 \
  --stream
```

### 2. `vajra-lm chat`
Start an interactive chat REPL session.
```bash
vajra-lm chat --config configs/training/pretrain_tiny.yaml
```

### 3. `vajra-lm evaluate`
Run model perplexity evaluation on tokenized datasets.
```bash
vajra-lm evaluate \
  --config configs/training/pretrain_tiny.yaml \
  --data_dir data/tokenized \
  --output evaluation_report.json
```

### 4. `vajra-lm tokenize`
Tokenize or detokenize text.
```bash
# Tokenize
vajra-lm tokenize --text "Hello world" --tokenizer tokenizer/v1.0

# Detokenize
vajra-lm tokenize --text "[12, 45, 88]" --decode --tokenizer tokenizer/v1.0
```

### 5. `vajra-lm profile`
Run training and inference benchmarks.
```bash
vajra-lm profile --config configs/training/pretrain_tiny.yaml --precision fp32
```
