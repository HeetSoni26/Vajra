from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import yaml


@dataclass(frozen=True)
class ModelConfig:
    model_name: str = "vajra-lm"
    vocab_size: int = 65536
    hidden_size: int = 2048
    num_layers: int = 24
    num_attention_heads: int = 16
    num_key_value_heads: int = 8
    intermediate_size: int = 5632
    max_position_embeddings: int = 4096
    rope_theta: float = 10000.0
    rms_norm_eps: float = 1e-6
    tie_word_embeddings: bool = True
    use_flash_attention: bool = True
    use_gradient_checkpointing: bool = False

    @property
    def head_dim(self) -> int:
        if self.hidden_size % self.num_attention_heads != 0:
            raise ValueError("hidden_size must be divisible by num_attention_heads")
        return self.hidden_size // self.num_attention_heads

    @classmethod
    def from_yaml(cls, path: str | Path) -> "ModelConfig":
        data = yaml.safe_load(Path(path).read_text())
        return cls(**data)
