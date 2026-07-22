import os
import uuid
import numpy as np
from pathlib import Path
from typing import List
from dataset.sharding.models import ShardFormatConfig, ShardStatistics
from dataset.sharding.metadata import BinaryShardMetadata
from dataset.mixture.models import DatasetMixture

class BinaryShardWriter:
    """
    Writes fixed-length sequences natively to numpy memory maps or raw binary arrays.
    Handles automatic rotation when limits are reached.
    """
    def __init__(self, config: ShardFormatConfig, mixture: DatasetMixture, stats: ShardStatistics, vocab_size: int):
        self.config = config
        self.mixture = mixture
        self.stats = stats
        self.vocab_size = vocab_size
        
        self.output_dir = Path(config.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self._current_shard_id = str(uuid.uuid4())
        self._current_path = self.output_dir / f"{self._current_shard_id}.bin"
        self._current_file = open(self._current_path, 'wb')
        self._tokens_in_current_shard = 0
        self._sequences_in_current_shard = 0
        
        self.dtype_map = {
            "uint16": np.uint16,
            "uint32": np.uint32
        }
        self.np_dtype = self.dtype_map.get(self.config.dtype, np.uint16)

    def write(self, sequence: List[int]):
        """
        Writes a single packed sequence to the binary stream.
        """
        arr = np.array(sequence, dtype=self.np_dtype)
        self._current_file.write(arr.tobytes())
        
        self._tokens_in_current_shard += len(sequence)
        self._sequences_in_current_shard += 1
        
        if self._tokens_in_current_shard >= self.config.tokens_per_shard:
            self._rotate()

    def _rotate(self):
        """
        Closes current shard, generates metadata, and opens a new one.
        """
        if self._tokens_in_current_shard == 0:
            return
            
        self._current_file.close()
        
        checksum = ""
        if self.config.enable_checksum:
            checksum = BinaryShardMetadata.compute_file_checksum(self._current_path)
            
        metadata = BinaryShardMetadata(
            shard_id=self._current_shard_id,
            vocab_size=self.vocab_size,
            sequence_length=self.config.sequence_length,
            num_sequences=self._sequences_in_current_shard,
            num_tokens=self._tokens_in_current_shard,
            dtype=self.config.dtype,
            version=self.config.version,
            mixture_name=self.mixture.name,
            checksum=checksum
        )
        metadata.save(self.output_dir / f"{self._current_shard_id}.json")
        self.stats.total_shards_created += 1
        
        # Open next
        self._current_shard_id = str(uuid.uuid4())
        self._current_path = self.output_dir / f"{self._current_shard_id}.bin"
        self._current_file = open(self._current_path, 'wb')
        self._tokens_in_current_shard = 0
        self._sequences_in_current_shard = 0

    def close(self):
        self._rotate()
        # Clean up empty dangling files if present
        if hasattr(self, '_current_file') and not self._current_file.closed:
            self._current_file.close()
            
        if self._current_path.exists() and self._current_path.stat().st_size == 0:
            os.remove(self._current_path)
