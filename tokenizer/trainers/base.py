from abc import ABC, abstractmethod
from collections.abc import Iterable

from tokenizer.configs.settings import TokenizerConfig
from tokenizer.tokenizers.base import BaseTokenizer


class BaseTrainer(ABC):
    """
    Abstract interface for training tokenizers.
    """

    def __init__(self, config: TokenizerConfig):
        self.config = config

    @abstractmethod
    def train(self, document_iterator: Iterable[str]) -> BaseTokenizer:
        """
        Train a new tokenizer model from an iterator of strings.
        Returns the trained BaseTokenizer instance.
        """

    @abstractmethod
    def train_from_files(self, file_paths: list[str]) -> BaseTokenizer:
        """
        Train a new tokenizer model directly from files.
        """
