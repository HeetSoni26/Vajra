from enum import Enum

from pydantic import BaseModel, ConfigDict, Field


class TokenizerType(str, Enum):
    BPE = "bpe"
    BYTE_LEVEL_BPE = "byte_level_bpe"
    UNIGRAM = "unigram"
    WORDPIECE = "wordpiece"
    SENTENCEPIECE = "sentencepiece"
    CUSTOM = "custom"


class TokenizerConfig(BaseModel):
    """
    Configuration for tokenizer training and encoding.
    """

    tokenizer_type: TokenizerType = Field(default=TokenizerType.BPE)
    vocab_size: int = Field(default=32000)

    # Special Tokens
    bos_token: str = "<s>"
    eos_token: str = "</s>"
    unk_token: str = "<unk>"
    pad_token: str = "<pad>"
    additional_special_tokens: list[str] = Field(default_factory=list)

    # Pre-tokenization / Normalization
    enable_normalization: bool = True
    enable_pre_tokenization: bool = True
    character_coverage: float = Field(default=1.0)

    # Execution
    random_seed: int = 42
    max_workers: int = 4

    # I/O
    output_dir: str = "output/tokenizer"
    training_corpus_paths: list[str] = Field(default_factory=list)

    model_config = ConfigDict(use_enum_values=True)
