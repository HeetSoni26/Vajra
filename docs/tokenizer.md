# Vajra Tokenizer Subsystem Specification

[Overview](../README.md) | [Architecture](architecture.md) | [Dataset Pipeline](dataset_pipeline.md)

---

## Overview

The Vajra Tokenizer subsystem utilizes a custom Byte-Pair Encoding (BPE) vocabulary built on Hugging Face `tokenizers` library constructs. It features a large vocabulary size of `65,536` tokens tailored for multilingual web text, source code, mathematical markup, and special control tokens.

---

## Vocabulary & Special Tokens

| Token ID | Token String | Purpose |
| :--- | :--- | :--- |
| `0` | `<|pad|>` | Padding token for batch alignment |
| `1` | `<|bos|>` / `<|endoftext|>` | Beginning of Sequence |
| `2` | `<|eos|>` | End of Sequence |
| `3` | `<|unk|>` | Unknown token fallback |
| `4` | `<|im_start|>` | Conversation / Turn start |
| `5` | `<|im_end|>` | Conversation / Turn end |

---

## Training a Custom BPE Tokenizer

To train a new vocabulary on raw dataset text files:

```bash
python -m tokenizer.train_tokenizer \
    --input-files data/raw/corpus.txt \
    --vocab-size 65536 \
    --output-dir tokenizer/v1.0
```

---

## Python Tokenizer Usage

```python
from tokenizer.tokenization_vajra import VajraTokenizer

# Load tokenizer from directory
tokenizer = VajraTokenizer.from_pretrained("tokenizer/v1.0")

# Encode text to tokens
text = "Vajra foundation model pretraining"
input_ids = tokenizer.encode(text)
print("Encoded IDs:", input_ids)

# Decode tokens to text
decoded_text = tokenizer.decode(input_ids)
print("Decoded Text:", decoded_text)
```
