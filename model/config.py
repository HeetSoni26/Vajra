import json
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, model_validator


class VajraConfig(BaseModel):
    model_config = ConfigDict(extra="ignore", populate_by_name=True)

    vocab_size: int = 102400
    hidden_size: int = 2048
    intermediate_size: int = 8192
    num_layers: int = 22
    num_attention_heads: int = 32
    num_key_value_heads: int = 8  # GQA
    max_position_embeddings: int = 4096
    rope_theta: float = 10000.0
    dropout: float = 0.0
    attention_dropout: float = 0.0
    rmsnorm_eps: float = 1e-5
    attention_bias: bool = False
    tie_word_embeddings: bool = False
    activation_function: str = "swiglu"
    dtype: str = "bfloat16"
    device: str = "cpu"
    use_gradient_checkpointing: bool = False
    model_name: str = "vajra"

    @model_validator(mode="before")
    @classmethod
    def handle_aliases(cls, data: Any) -> Any:
        if isinstance(data, dict):
            data = dict(data)
            if "rms_norm_eps" in data and "rmsnorm_eps" not in data:
                data["rmsnorm_eps"] = data["rms_norm_eps"]
            if "num_hidden_layers" in data and "num_layers" not in data:
                data["num_layers"] = data["num_hidden_layers"]
        return data

    @property
    def rms_norm_eps(self) -> float:
        return self.rmsnorm_eps

    @rms_norm_eps.setter
    def rms_norm_eps(self, value: float):
        self.rmsnorm_eps = value

    @property
    def head_dim(self) -> int:
        return self.hidden_size // self.num_attention_heads

    @classmethod
    def from_pretrained(cls, path: Path | str) -> "VajraConfig":
        path = Path(path)
        config_file = path / "config.json" if path.is_dir() else path
        with open(config_file, "r", encoding="utf-8") as f:
            return cls.model_validate(json.load(f))

    def save_pretrained(self, path: Path | str):
        path = Path(path)
        if path.suffix != ".json":
            path.mkdir(parents=True, exist_ok=True)
            config_file = path / "config.json"
        else:
            path.parent.mkdir(parents=True, exist_ok=True)
            config_file = path
        with open(config_file, "w", encoding="utf-8") as f:
            f.write(self.model_dump_json(indent=2))

    @classmethod
    def from_yaml(cls, path: Path | str) -> "VajraConfig":
        import yaml

        path = Path(path)
        with open(path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)
        return cls.model_validate(data)

    def to_yaml(self, path: Path | str):
        import yaml

        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            yaml.safe_dump(self.model_dump(), f)


# Canonical Alias for Backward Compatibility
ModelConfig = VajraConfig


def get_preset(name: str) -> VajraConfig:
    presets = {
        "Vajra-370M": VajraConfig(
            vocab_size=102400,
            hidden_size=1024,
            intermediate_size=4096,
            num_layers=24,
            num_attention_heads=16,
            num_key_value_heads=4,
        ),
        "Vajra-1B": VajraConfig(
            vocab_size=102400,
            hidden_size=2048,
            intermediate_size=8192,
            num_layers=22,
            num_attention_heads=32,
            num_key_value_heads=8,
        ),
        "Vajra-3B": VajraConfig(
            vocab_size=102400,
            hidden_size=3072,
            intermediate_size=12288,
            num_layers=32,
            num_attention_heads=32,
            num_key_value_heads=8,
        ),
        "Vajra-7B": VajraConfig(
            vocab_size=102400,
            hidden_size=4096,
            intermediate_size=14336,
            num_layers=32,
            num_attention_heads=32,
            num_key_value_heads=8,
        ),
    }
    if name not in presets:
        raise ValueError(f"Preset {name} not found.")
    return presets[name]
