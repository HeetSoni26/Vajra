from abc import ABC, abstractmethod

from tokenizer.configs.settings import TokenizerConfig


class BaseTokenizer(ABC):
    """
    Abstract base interface for all Vajra tokenizers.
    Future implementations (BPE, SentencePiece, etc.) must subclass this.
    """

    def __init__(self, config: TokenizerConfig):
        self.config = config

    @abstractmethod
    def encode(self, text: str) -> list[int]:
        """Convert a string to a list of token IDs."""

    @abstractmethod
    def decode(self, ids: list[int]) -> str:
        """Convert a list of token IDs back to a string."""

    @abstractmethod
    def get_vocab_size(self) -> int:
        """Return the size of the vocabulary."""

    @abstractmethod
    def save_pretrained(self, save_directory: str) -> None:
        """Save the tokenizer model and configurations."""

    @classmethod
    @abstractmethod
    def from_pretrained(cls, save_directory: str) -> "BaseTokenizer":
        """Load a tokenizer from a directory."""
