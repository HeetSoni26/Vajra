from pydantic import BaseModel, Field

class ShardMetadata(BaseModel):
    """
    Metadata for a binary token shard.
    """
    shard_id: str = Field(..., description="Unique identifier for the shard.")
    vocab_size: int = Field(..., description="Vocabulary size used for tokenization.")
    total_tokens: int = Field(..., description="Total tokens in this shard.")
    sequence_length: int = Field(..., description="Fixed sequence length (if packed).")
    dtype: str = Field(default="uint16", description="Data type of the tokens (e.g., uint16, uint32).")
    version: str = Field(default="1.0", description="Shard format version.")
