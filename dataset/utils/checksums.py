import hashlib
from pathlib import Path

def calculate_sha256(filepath: str | Path, chunk_size: int = 8192) -> str:
    """
    Calculates the SHA256 checksum of a file efficiently.
    """
    sha256_hash = hashlib.sha256()
    path = Path(filepath)
    
    if not path.exists():
        raise FileNotFoundError(f"File not found for checksum: {filepath}")
        
    with open(path, "rb") as f:
        for byte_block in iter(lambda: f.read(chunk_size), b""):
            sha256_hash.update(byte_block)
            
    return sha256_hash.hexdigest()
