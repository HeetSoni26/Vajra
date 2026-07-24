from typing import List
from tokenizer.tokenizers.base import BaseTokenizer
from tokenizer.vocab.manager import VocabularyManager


class TokenizerValidator:
    """
    Validates tokenizers for consistency, round-trip encoding, and vocabulary rules.
    """

    def __init__(self, tokenizer: BaseTokenizer, vocab_manager: VocabularyManager):
        self.tokenizer = tokenizer
        self.vocab_manager = vocab_manager

    def validate_round_trip(self, text: str) -> bool:
        """
        Validates that text -> encode -> decode -> text is functionally equivalent.
        (Note: depending on normalization, it might not be string-exact).
        """
        encoded = self.tokenizer.encode(text)
        decoded = self.tokenizer.decode(encoded)
        return text.strip() == decoded.strip()

    def validate_special_tokens(self, special_tokens: List[str]) -> bool:
        """
        Ensures all required special tokens are in the vocabulary.
        """
        for token in special_tokens:
            if token not in self.vocab_manager.token_to_id:
                return False
        return True
