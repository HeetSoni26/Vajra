from typing import Iterable, List, Iterator
from tokenizer.tokenizers.base import BaseTokenizer

class TokenizationPipeline:
    """
    Manages the bulk tokenization of prepared documents.
    """
    def __init__(self, tokenizer: BaseTokenizer):
        self.tokenizer = tokenizer
        
    def encode_stream(self, text_stream: Iterable[str]) -> Iterator[List[int]]:
        """
        Lazily yields tokenized sequences from a text stream.
        """
        for text in text_stream:
            yield self.tokenizer.encode(text)
            
    def encode_batch(self, texts: List[str]) -> List[List[int]]:
        """
        Encodes a batch of texts in memory.
        """
        return [self.tokenizer.encode(text) for text in texts]
