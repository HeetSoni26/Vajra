import tempfile
from pathlib import Path

import numpy as np

from dataset.mixture.models import DatasetMixture
from dataset.sharding.metadata import BinaryShardMetadata
from dataset.sharding.models import ShardFormatConfig, ShardStatistics
from dataset.sharding.packer import SequencePackingEngine
from dataset.sharding.pipeline import ShardingPipeline
from dataset.sharding.reader import BinaryShardReader
from dataset.sharding.validators import ShardValidator
from dataset.sharding.writer import BinaryShardWriter
from tokenizer.configs.settings import TokenizerConfig
from tokenizer.tokenizers.mock import MockTokenizer


def test_sequence_packer():
    config = ShardFormatConfig(sequence_length=10)
    tok_config = TokenizerConfig(bos_token="1", eos_token="2", pad_token="0")
    tokenizer = MockTokenizer(tok_config)

    # We manually override these for mock tokenizer so it returns ints
    # Note: the packer tries to access token_to_id on hfbpe, but falls back to 1,2,0 for mock

    stats = ShardStatistics()
    packer = SequencePackingEngine(tokenizer, config, stats)

    stream = [[3, 4, 5], [6, 7, 8, 9]]

    # Pack:
    # [1, 3, 4, 5, 2] -> len 5
    # [1, 6, 7, 8, 9, 2] -> len 6
    # Total = 11. Chunk 1 = 10, leftover = 1
    chunks = list(packer.pack(iter(stream)))
    assert len(chunks) == 1
    assert len(chunks[0]) == 10

    # Flush
    flush_chunks = list(packer.flush())
    assert len(flush_chunks) == 1
    assert len(flush_chunks[0]) == 10
    assert stats.total_padding_tokens == 9


def test_shard_writer_and_reader():
    mixture = DatasetMixture(name="test_mix")
    config = ShardFormatConfig(sequence_length=4, tokens_per_shard=8, dtype="uint16")
    stats = ShardStatistics()

    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as d:
        config.output_dir = d
        writer = BinaryShardWriter(config, mixture, stats, vocab_size=100)

        # Write 2 sequences (8 tokens), should rotate
        writer.write([1, 2, 3, 4])
        writer.write([5, 6, 7, 8])

        # Write 1 sequence (4 tokens), should be in next shard
        writer.write([9, 10, 11, 12])
        writer.close()

        assert stats.total_shards_created == 2

        # List json files
        meta_files = list(Path(d).glob("*.json"))
        assert len(meta_files) == 2

        # Find the one with 8 tokens
        meta1 = next(m for m in meta_files if BinaryShardMetadata.load(m).num_tokens == 8)
        reader = BinaryShardReader(meta1)

        assert reader.metadata.num_sequences == 2
        assert reader.verify_integrity() is True

        seqs = list(reader.stream())
        assert len(seqs) == 2
        np.testing.assert_array_equal(seqs[0], np.array([1, 2, 3, 4], dtype=np.uint16))
        np.testing.assert_array_equal(seqs[1], np.array([5, 6, 7, 8], dtype=np.uint16))

        # Force garbage collection of memmap to unlock file on Windows
        import gc

        del seqs
        del reader
        gc.collect()


def test_pipeline_integration():
    tok_config = TokenizerConfig()
    tokenizer = MockTokenizer(tok_config)
    mixture = DatasetMixture(name="test_mix")
    config = ShardFormatConfig(sequence_length=5, tokens_per_shard=20)

    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as d:
        config.output_dir = d
        pipeline = ShardingPipeline(tokenizer, mixture, config)

        text_stream = ["hello world", "this is vajra test"]
        stats = pipeline.execute(text_stream)

        assert stats.total_documents_processed == 2

        meta_files = list(Path(d).glob("*.json"))
        assert len(meta_files) >= 1

        import gc

        gc.collect()


def test_validator():
    mixture = DatasetMixture(name="test_mix")
    config = ShardFormatConfig(sequence_length=4, tokens_per_shard=8, dtype="uint16")
    stats = ShardStatistics()

    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as d:
        config.output_dir = d
        writer = BinaryShardWriter(config, mixture, stats, vocab_size=10)  # Vocab 10
        writer.write([1, 2, 3, 99])  # Out of bounds!
        writer.close()

        meta_file = next(Path(d).glob("*.json"))
        res = ShardValidator.validate_shard(meta_file)

        assert res["valid"] is False
        assert any("out-of-bounds" in e for e in res["errors"])

        import gc

        gc.collect()
