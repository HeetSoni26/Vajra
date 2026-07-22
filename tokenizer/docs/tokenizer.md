# Vajra Tokenizer Framework & Training Infrastructure

The Tokenizer Framework provides the core abstraction layers and utilities for training, encoding, validating, and generating binary token shards within the Vajra ecosystem. 

## Design Philosophy
This module is strictly decoupled from any specific dataset downloaders or PyTorch `DataLoader` logic. It exists exclusively to bridge the gap between cleaned document text and packed binary IDs.

## Core Modules
- **`configs/`**: Exposes the highly extensible `TokenizerConfig` which orchestrates vocabulary size, special token injection, and normalization toggles.
- **`tokenizers/`**: Base interface `BaseTokenizer` providing a unified contract (`encode`, `decode`, `get_vocab_size`) that subsequent implementations (BPE, SentencePiece) will adhere to.
- **`trainers/`**: Base interface `BaseTrainer` allowing pluggable integration with Hugging Face Tokenizers or Google SentencePiece training backends.
- **`vocab/`**: High-performance, JSON-serializable `VocabularyManager` for maintaining state and statistics on vocabulary mappings.
- **`validators/`**: `TokenizerValidator` guarantees data integrity via round-trip consistency checking and special token validation.
- **`encoders/`**: `TokenizationPipeline` manages parallelization and chunking of text streams.
- **`shards/`**: Provides the structural blueprints (`ShardMetadata`, `BaseShardWriter`, `BaseShardReader`) for saving massive integer arrays into optimized contiguous memory formats.
- **`statistics/`**: `TokenizerStatistics` exposes analytical insights (compression ratio, unknown frequency).

## CLI Operations
The `manage_tokenizer.py` interface provides:
- `train`: Spin up a trainer backend.
- `validate`: Check tokenizer encoding integrity.
- `encode` / `decode`: Quick testing utilities.
- `stats`: Print vocabulary statistics.

*Note: As per Milestone 4 constraints, the CLI currently implements mock handlers for infrastructural validation.*
