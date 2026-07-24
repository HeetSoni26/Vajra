from typing import Iterable, List
from tokenizer.trainers.base import BaseTrainer
from tokenizer.tokenizers.hf_bpe import HFBpeTokenizer
from tokenizer.tokenizers.base import BaseTokenizer
from tokenizer.configs.settings import TokenizerConfig

try:
    from tokenizers.trainers import BpeTrainer
except ImportError:
    pass


class HFBpeTrainer(BaseTrainer):
    """
    Trainer for the HFBpeTokenizer backend.
    """

    def __init__(self, config: TokenizerConfig):
        super().__init__(config)

    def _create_hf_trainer(self):
        special_tokens = [
            self.config.unk_token,
            self.config.bos_token,
            self.config.eos_token,
            self.config.pad_token,
        ] + self.config.additional_special_tokens

        # Ensures no duplicates if pad/unk are same
        special_tokens = list(dict.fromkeys(special_tokens))

        return BpeTrainer(
            vocab_size=self.config.vocab_size,
            min_frequency=2,
            special_tokens=special_tokens,
            show_progress=True,
        )

    def train(self, document_iterator: Iterable[str]) -> BaseTokenizer:
        tokenizer = HFBpeTokenizer(self.config)
        trainer = self._create_hf_trainer()

        tokenizer._tokenizer.train_from_iterator(
            document_iterator,
            trainer=trainer,
            length=None,  # If iterator has unknown length
        )

        # Fix the post processor IDs now that vocab is built
        bos_id = tokenizer._tokenizer.token_to_id(self.config.bos_token)
        eos_id = tokenizer._tokenizer.token_to_id(self.config.eos_token)

        from tokenizers.processors import TemplateProcessing

        tokenizer._tokenizer.post_processor = TemplateProcessing(
            single=f"{self.config.bos_token} $A {self.config.eos_token}",
            pair=f"{self.config.bos_token} $A {self.config.eos_token} $B:1 {self.config.eos_token}:1",
            special_tokens=[(self.config.bos_token, bos_id), (self.config.eos_token, eos_id)],
        )

        return tokenizer

    def train_from_files(self, file_paths: List[str]) -> BaseTokenizer:
        tokenizer = HFBpeTokenizer(self.config)
        trainer = self._create_hf_trainer()

        tokenizer._tokenizer.train(files=file_paths, trainer=trainer)

        # Re-initialize template processing with correct IDs
        bos_id = tokenizer._tokenizer.token_to_id(self.config.bos_token)
        eos_id = tokenizer._tokenizer.token_to_id(self.config.eos_token)

        from tokenizers.processors import TemplateProcessing

        tokenizer._tokenizer.post_processor = TemplateProcessing(
            single=f"{self.config.bos_token} $A {self.config.eos_token}",
            pair=f"{self.config.bos_token} $A {self.config.eos_token} $B:1 {self.config.eos_token}:1",
            special_tokens=[(self.config.bos_token, bos_id), (self.config.eos_token, eos_id)],
        )

        return tokenizer
