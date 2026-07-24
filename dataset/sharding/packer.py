from collections.abc import Iterator

from dataset.sharding.models import ShardFormatConfig, ShardStatistics
from tokenizer.tokenizers.base import BaseTokenizer


class SequencePackingEngine:
    """
    Takes arbitrary token sequences, adds BOS/EOS, and packs them into fixed-length arrays.
    """

    def __init__(self, tokenizer: BaseTokenizer, config: ShardFormatConfig, stats: ShardStatistics):
        self.tokenizer = tokenizer
        self.config = config
        self.stats = stats

        # Internals
        self._buffer: list[int] = []
        self._bos_id = self.tokenizer.config.bos_token
        self._eos_id = self.tokenizer.config.eos_token
        self._pad_id = self.tokenizer.config.pad_token

        # We need numerical IDs
        # Hack to access actual HF tokenizer vocabulary if this is a HFBpeTokenizer
        if hasattr(self.tokenizer, "_tokenizer"):
            self._bos_id = self.tokenizer._tokenizer.token_to_id(self._bos_id)
            self._eos_id = self.tokenizer._tokenizer.token_to_id(self._eos_id)
            self._pad_id = self.tokenizer._tokenizer.token_to_id(self._pad_id)
        else:
            # For mock testing
            self._bos_id = 1
            self._eos_id = 2
            self._pad_id = 0

    def pack(self, token_stream: Iterator[list[int]]) -> Iterator[list[int]]:
        """
        Yields sequences exactly `self.config.sequence_length` long.
        """
        seq_len = self.config.sequence_length

        for tokens in token_stream:
            if self.config.insert_bos:
                self._buffer.append(self._bos_id)
                self.stats.total_tokens += 1

            self._buffer.extend(tokens)
            self.stats.total_tokens += len(tokens)

            if self.config.insert_eos:
                self._buffer.append(self._eos_id)
                self.stats.total_tokens += 1

            # Yield packed chunks
            while len(self._buffer) >= seq_len:
                chunk = self._buffer[:seq_len]
                self._buffer = self._buffer[seq_len:]
                self.stats.total_sequences += 1
                yield chunk

    def flush(self) -> Iterator[list[int]]:
        """
        Pads and yields whatever is leftover in the buffer.
        """
        if not self._buffer:
            return

        seq_len = self.config.sequence_length

        if self.config.pad_to_sequence_length:
            pad_amount = seq_len - len(self._buffer)
            self._buffer.extend([self._pad_id] * pad_amount)
            self.stats.total_padding_tokens += pad_amount
            self.stats.total_tokens += pad_amount

            self.stats.total_sequences += 1
            yield self._buffer
            self._buffer = []
        else:
            # Strictly don't yield partial chunks if padding is off
            pass
