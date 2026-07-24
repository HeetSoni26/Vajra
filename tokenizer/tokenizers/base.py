from abc import ABC, abstractmethod
from typing import List
from tokenizer.configs.settings import TokenizerConfig


class BaseTokenizer(ABC):
    """
    Abstract base interface for all Vajra tokenizers.
    Future implementations (BPE, SentencePiece, etc.) must subclass this.
    """

    def __init__(self, config: TokenizerConfig):
        self.config = config

    @abstractmethod
    def encode(self, text: str) -> List[int]:
        """Convert a string to a list of token IDs."""
        pass

    @abstractmethod
    def decode(self, ids: List[int]) -> str:
        """Convert a list of token IDs back to a string."""
        pass

    @abstractmethod
    def get_vocab_size(self) -> int:
        """Return the size of the vocabulary."""
        pass

    @abstractmethod
    def save_pretrained(self, save_directory: str) -> None:
        """Save the tokenizer model and configurations."""
        pass

    @classmethod
    @abstractmethod
    def from_pretrained(cls, save_directory: str) -> "BaseTokenizer":
        """Load a tokenizer from a directory."""
        pass
