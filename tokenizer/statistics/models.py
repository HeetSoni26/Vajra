from pydantic import BaseModel

class TokenizerStatistics(BaseModel):
    """
    Statistics generated during or after tokenization.
    """
    vocab_size: int = 0
    total_documents: int = 0
    total_characters: int = 0
    total_tokens: int = 0
    unknown_tokens_count: int = 0
    special_tokens_count: int = 0
    
    @property
    def compression_ratio(self) -> float:
        """Ratio of characters per token. Higher is better (usually)."""
        if self.total_tokens == 0:
            return 0.0
        return self.total_characters / self.total_tokens
        
    @property
    def unknown_token_frequency(self) -> float:
        if self.total_tokens == 0:
            return 0.0
        return self.unknown_tokens_count / self.total_tokens
