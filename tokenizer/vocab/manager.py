import json
from pathlib import Path
from typing import Dict
from tokenizer.configs.settings import TokenizerConfig


class VocabularyManager:
    """
    Manages the tokenizer vocabulary mapping, metadata, and versioning.
    """

    def __init__(self, config: TokenizerConfig):
        self.config = config
        self.token_to_id: Dict[str, int] = {}
        self.id_to_token: Dict[int, str] = {}

    def add_token(self, token: str) -> int:
        if token in self.token_to_id:
            return self.token_to_id[token]
        token_id = len(self.token_to_id)
        self.token_to_id[token] = token_id
        self.id_to_token[token_id] = token
        return token_id

    def load(self, path: str | Path) -> None:
        """Loads vocabulary from a JSON file."""
        with open(path, "r", encoding="utf-8") as f:
            self.token_to_id = json.load(f)
            self.id_to_token = {v: k for k, v in self.token_to_id.items()}

    def save(self, path: str | Path) -> None:
        """Saves vocabulary to a JSON file."""
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.token_to_id, f, ensure_ascii=False, indent=2)

    def get_statistics(self) -> dict:
        return {
            "vocab_size": len(self.token_to_id),
            "special_tokens": [
                self.config.bos_token,
                self.config.eos_token,
                self.config.pad_token,
                self.config.unk_token,
            ]
            + self.config.additional_special_tokens,
        }
