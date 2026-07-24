from pydantic import BaseModel, Field


class ShardFormatConfig(BaseModel):
    """
    Configuration for sequence packing and binary shard formatting.
    """

    sequence_length: int = Field(
        default=2048, description="Fixed token length for all sequences in the shard."
    )
    tokens_per_shard: int = Field(
        default=100_000_000, description="Target number of tokens per binary file chunk."
    )
    dtype: str = Field(default="uint16", description="Numpy data type (uint16 or uint32).")

    # Packing strategies
    pad_to_sequence_length: bool = Field(
        default=True, description="Whether to pad final sequences."
    )
    insert_bos: bool = Field(
        default=True, description="Insert beginning of sequence token per document."
    )
    insert_eos: bool = Field(default=True, description="Insert end of sequence token per document.")

    # Checksum & Verification
    enable_checksum: bool = Field(
        default=True, description="Generate MD5 verification checksum per shard."
    )
    version: str = Field(default="1.0", description="Format version.")

    output_dir: str = Field(default="output/shards")


class ShardStatistics(BaseModel):
    total_documents_processed: int = 0
    total_tokens: int = 0
    total_sequences: int = 0
    total_padding_tokens: int = 0
    total_shards_created: int = 0

    @property
    def padding_percentage(self) -> float:
        if self.total_tokens == 0:
            return 0.0
        return (self.total_padding_tokens / self.total_tokens) * 100.0
