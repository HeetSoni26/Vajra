# Vajra Dataset Tokenization & Binary Sharding

The Sharding framework serves as the final terminal output layer of the Dataset Module. It successfully bridges prepared streaming text, dynamic sequence tokenization via the HuggingFace backend, and writes the output continuously into dense `numpy.memmap` compatible binary chunks.

## Architecture

- **`models.py`**: Exports `ShardFormatConfig` controlling target sequence padding, sequence boundaries, and physical file rotation targets (`tokens_per_shard`).
- **`metadata.py`**: Defines `BinaryShardMetadata`, mapping physical binary payloads back to versioning checksums, mixtures, and integer typing limits (e.g. `uint16`).
- **`packer.py`**: Exports `SequencePackingEngine`, dynamically appending configurable `<bos>` and `<eos>` tags directly to raw text tokens, chunking arrays to exactly `[sequence_length]` dimensions, and carrying leftover buffers safely across document gaps.
- **`writer.py`**: `BinaryShardWriter` tracks byte limits per session. It automatically dumps `ShardMetadata.json` logs before safely closing the chunk and initiating rotation immediately back onto the stream payload.
- **`reader.py`**: Exposes `BinaryShardReader` mapping arrays back locally into zero-copy memory arrays utilizing `numpy.memmap` logic.
- **`pipeline.py`**: Connects dataset pipelines with `ShardingPipeline`, providing an explicit 1:1 map mapping tokenization over the iteration limits without retaining data directly in RAM overhead.
- **`validators.py`**: Exposes `ShardValidator`, forcing binary payload constraints, protecting out of bounds token mapping exceptions prior to tensor operations.

## CLI Commands

The module natively plugs into `manage_dataset.py`:

```bash
# Compile and stream chunks utilizing the designated mixture parameters:
python dataset/scripts/manage_dataset.py shard build --output-dir "output/shards"

# Randomly verify output bounds limit mapping integrity
python dataset/scripts/manage_dataset.py shard verify output/shards/some_shard.json

# Read checksum configuration structure
python dataset/scripts/manage_dataset.py shard inspect output/shards/some_shard.json
```
