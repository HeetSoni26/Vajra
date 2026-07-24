import os
from typing import Optional

from tokenizer.configs.settings import TokenizerConfig
from tokenizer.tokenizers.base import BaseTokenizer

try:
    from tokenizers import Tokenizer
    from tokenizers.models import BPE
    from tokenizers.normalizers import NFD, Sequence, StripAccents
    from tokenizers.pre_tokenizers import Whitespace
    from tokenizers.processors import TemplateProcessing
except ImportError:
    pass  # Allow offline linting if missing


class HFBpeTokenizer(BaseTokenizer):
    """
    A production-grade BPE tokenizer backend utilizing the Rust-backed Hugging Face tokenizers library.
    """

    def __init__(self, config: TokenizerConfig, hf_tokenizer: Optional["Tokenizer"] = None):
        super().__init__(config)

        if hf_tokenizer is not None:
            self._tokenizer = hf_tokenizer
        else:
            self._tokenizer = Tokenizer(BPE(unk_token=config.unk_token))
            self._setup_pipeline()

    def _setup_pipeline(self):
        # 1. Normalization
        if self.config.enable_normalization:
            self._tokenizer.normalizer = Sequence([NFD(), StripAccents()])

        # 2. Pre-tokenization
        if self.config.enable_pre_tokenization:
            self._tokenizer.pre_tokenizer = Whitespace()

        # 3. Post-processing (Template)
        self._tokenizer.post_processor = TemplateProcessing(
            single=f"{self.config.bos_token} $A {self.config.eos_token}",
            pair=f"{self.config.bos_token} $A {self.config.eos_token} $B:1 {self.config.eos_token}:1",
            special_tokens=[
                (
                    self.config.bos_token,
                    self.config.vocab_size + 1,
                ),  # Placeholder IDs, overridden on train
                (self.config.eos_token, self.config.vocab_size + 2),
            ],
        )

    def encode(self, text: str) -> list[int]:
        encoded = self._tokenizer.encode(text)
        return encoded.ids

    def decode(self, ids: list[int]) -> str:
        return self._tokenizer.decode(ids, skip_special_tokens=False)

    def get_vocab_size(self) -> int:
        return self._tokenizer.get_vocab_size()

    def save_pretrained(self, save_directory: str) -> None:
        os.makedirs(save_directory, exist_ok=True)
        path = os.path.join(save_directory, "tokenizer.json")
        self._tokenizer.save(path)

    @classmethod
    def from_pretrained(cls, save_directory: str) -> "HFBpeTokenizer":
        path = os.path.join(save_directory, "tokenizer.json")
        if not os.path.exists(path):
            raise FileNotFoundError(f"Tokenizer not found at {path}")

        hf_tokenizer = Tokenizer.from_file(path)
        config = TokenizerConfig()  # Ideally load config.json here too

        return cls(config=config, hf_tokenizer=hf_tokenizer)
