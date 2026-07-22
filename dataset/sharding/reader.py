import numpy as np
from pathlib import Path
from typing import Iterator
from dataset.sharding.metadata import BinaryShardMetadata

class BinaryShardReader:
    """
    Reads natively packed binary arrays from disk iteratively.
    """
    def __init__(self, metadata_path: str | Path):
        self.metadata_path = Path(metadata_path)
        self.metadata = BinaryShardMetadata.load(self.metadata_path)
        
        self.binary_path = self.metadata_path.with_suffix('.bin')
        if not self.binary_path.exists():
            raise FileNotFoundError(f"Binary tensor not found for metadata: {self.binary_path}")
            
        self.dtype_map = {
            "uint16": np.uint16,
            "uint32": np.uint32
        }
        self.np_dtype = self.dtype_map.get(self.metadata.dtype, np.uint16)

    def verify_integrity(self) -> bool:
        """
        Validates checksum if present in metadata.
        """
        if not self.metadata.checksum:
            return True
            
        file_hash = BinaryShardMetadata.compute_file_checksum(self.binary_path)
        return file_hash == self.metadata.checksum

    def stream(self) -> Iterator[np.ndarray]:
        """
        Yields numpy arrays of shape `[sequence_length]`.
        Uses memory mapping for zero-copy streaming capability.
        """
        # Memory map the binary file
        memmap = np.memmap(
            self.binary_path, 
            dtype=self.np_dtype, 
            mode='r',
            shape=(self.metadata.num_sequences, self.metadata.sequence_length)
        )
        
        for i in range(self.metadata.num_sequences):
            yield memmap[i]
