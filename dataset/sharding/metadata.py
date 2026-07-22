import json
import hashlib
from pathlib import Path
from pydantic import BaseModel

class BinaryShardMetadata(BaseModel):
    """
    Metadata linked to a physical binary tensor file on disk.
    """
    shard_id: str
    vocab_size: int
    sequence_length: int
    num_sequences: int
    num_tokens: int
    dtype: str
    version: str
    mixture_name: str
    checksum: str = ""
    compression: str = "none"
    
    def save(self, path: Path):
        with open(path, 'w', encoding='utf-8') as f:
            f.write(self.model_dump_json(indent=2))
            
    @classmethod
    def load(cls, path: Path) -> 'BinaryShardMetadata':
        with open(path, 'r', encoding='utf-8') as f:
            return cls.model_validate(json.load(f))
            
    @staticmethod
    def compute_file_checksum(filepath: Path) -> str:
        md5 = hashlib.md5()
        with open(filepath, 'rb') as f:
            for chunk in iter(lambda: f.read(4096), b""):
                md5.update(chunk)
        return md5.hexdigest()
