from abc import ABC, abstractmethod
from collections.abc import Iterator

from tokenizer.shards.metadata import ShardMetadata


class BaseShardReader(ABC):
    """
    Interface for reading token IDs from binary shards on disk.
    """

    def __init__(self, input_path: str):
        self.input_path = input_path
        self.metadata = self._load_metadata()

    @abstractmethod
    def _load_metadata(self) -> ShardMetadata:
        pass

    @abstractmethod
    def stream_sequences(self) -> Iterator[list[int]]:
        """Yields token sequences from the shard."""
