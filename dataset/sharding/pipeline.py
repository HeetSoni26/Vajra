from collections.abc import Iterable

from dataset.mixture.models import DatasetMixture
from dataset.sharding.models import ShardFormatConfig, ShardStatistics
from dataset.sharding.packer import SequencePackingEngine
from dataset.sharding.writer import BinaryShardWriter
from tokenizer.tokenizers.base import BaseTokenizer


class ShardingPipeline:
    """
    Connects raw text stream -> Tokenization -> Sequence Packing -> Shard Writing.
    """

    def __init__(
        self, tokenizer: BaseTokenizer, mixture: DatasetMixture, config: ShardFormatConfig
    ):
        self.tokenizer = tokenizer
        self.mixture = mixture
        self.config = config
        self.stats = ShardStatistics()

    def execute(self, text_stream: Iterable[str]) -> ShardStatistics:
        """
        Executes the entire sharding pipeline on a streamed generator of text.
        """

        # We need an intermediate generator for encoding
        def _encode_stream(texts: Iterable[str]):
            for t in texts:
                self.stats.total_documents_processed += 1
                yield self.tokenizer.encode(t)

        packer = SequencePackingEngine(self.tokenizer, self.config, self.stats)
        writer = BinaryShardWriter(
            self.config, self.mixture, self.stats, self.tokenizer.get_vocab_size()
        )

        # Stream text -> Tokens -> Packed -> Binary
        packed_stream = packer.pack(_encode_stream(text_stream))

        for sequence in packed_stream:
            writer.write(sequence)

        # Flush dangling tokens
        for sequence in packer.flush():
            writer.write(sequence)

        writer.close()

        return self.stats
