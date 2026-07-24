from tokenizer.configs.settings import TokenizerConfig
from tokenizer.tokenizers.base import BaseTokenizer


class MockTokenizer(BaseTokenizer):
    """
    A simple mock tokenizer for testing the framework infrastructure.
    It splits text by spaces and builds a naive vocabulary.
    """

    def __init__(self, config: TokenizerConfig):
        super().__init__(config)
        self.vocab = {}
        self.inv_vocab = {}

    def encode(self, text: str) -> list[int]:
        ids = []
        for word in text.split():
            if word not in self.vocab:
                new_id = len(self.vocab)
                self.vocab[word] = new_id
                self.inv_vocab[new_id] = word
            ids.append(self.vocab[word])
        return ids

    def decode(self, ids: list[int]) -> str:
        return " ".join([self.inv_vocab.get(i, self.config.unk_token) for i in ids])

    def get_vocab_size(self) -> int:
        return len(self.vocab)

    def save_pretrained(self, save_directory: str) -> None:
        pass

    @classmethod
    def from_pretrained(cls, save_directory: str) -> "MockTokenizer":
        return cls(TokenizerConfig())
