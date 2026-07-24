from abc import ABC, abstractmethod
from typing import List
from tokenizer.shards.metadata import ShardMetadata


class BaseShardWriter(ABC):
    """
    Interface for writing token IDs to binary shards on disk.
    """

    def __init__(self, output_path: str, metadata: ShardMetadata):
        self.output_path = output_path
        self.metadata = metadata

    @abstractmethod
    def write_batch(self, token_batch: List[List[int]]) -> None:
        """Writes a batch of token sequences."""
        pass

    @abstractmethod
    def close(self) -> None:
        """Closes the shard and saves metadata."""
        pass
