import time
from typing import List
from tokenizer.tokenizers.base import BaseTokenizer

class TokenizerBenchmark:
    """
    Utilities for benchmarking tokenizer performance (speed, throughput).
    """
    def __init__(self, tokenizer: BaseTokenizer):
        self.tokenizer = tokenizer

    def benchmark_encoding(self, texts: List[str]) -> dict:
        """
        Benchmarks encoding speed on a list of strings.
        """
        start_time = time.perf_counter()
        total_chars = 0
        total_tokens = 0
        
        for text in texts:
            total_chars += len(text)
            encoded = self.tokenizer.encode(text)
            total_tokens += len(encoded)
            
        end_time = time.perf_counter()
        duration = end_time - start_time
        
        return {
            "duration_seconds": duration,
            "total_texts": len(texts),
            "total_chars": total_chars,
            "total_tokens": total_tokens,
            "chars_per_second": total_chars / duration if duration > 0 else 0,
            "tokens_per_second": total_tokens / duration if duration > 0 else 0,
            "compression_ratio": total_chars / total_tokens if total_tokens > 0 else 0
        }

    def benchmark_decoding(self, token_batches: List[List[int]]) -> dict:
        """
        Benchmarks decoding speed on a list of token lists.
        """
        start_time = time.perf_counter()
        total_tokens = 0
        
        for tokens in token_batches:
            total_tokens += len(tokens)
            self.tokenizer.decode(tokens)
            
        end_time = time.perf_counter()
        duration = end_time - start_time
        
        return {
            "duration_seconds": duration,
            "total_batches": len(token_batches),
            "total_tokens": total_tokens,
            "tokens_per_second": total_tokens / duration if duration > 0 else 0
        }
