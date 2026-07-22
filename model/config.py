import json
from pathlib import Path
from pydantic import BaseModel

class VajraConfig(BaseModel):
    vocab_size: int = 102400
    hidden_size: int = 2048
    intermediate_size: int = 8192
    num_layers: int = 22
    num_attention_heads: int = 32
    num_key_value_heads: int = 8 # GQA
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
    
    @classmethod
    def from_pretrained(cls, path: Path | str) -> 'VajraConfig':
        path = Path(path)
        with open(path / "config.json", "r") as f:
            return cls.model_validate(json.load(f))
            
    def save_pretrained(self, path: Path | str):
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        with open(path / "config.json", "w") as f:
            f.write(self.model_dump_json(indent=2))

def get_preset(name: str) -> VajraConfig:
    presets = {
        "Vajra-370M": VajraConfig(
            vocab_size=102400, hidden_size=1024, intermediate_size=4096,
            num_layers=24, num_attention_heads=16, num_key_value_heads=4
        ),
        "Vajra-1B": VajraConfig(
            vocab_size=102400, hidden_size=2048, intermediate_size=8192,
            num_layers=22, num_attention_heads=32, num_key_value_heads=8
        ),
        "Vajra-3B": VajraConfig(
            vocab_size=102400, hidden_size=3072, intermediate_size=12288,
            num_layers=32, num_attention_heads=32, num_key_value_heads=8
        ),
        "Vajra-7B": VajraConfig(
            vocab_size=102400, hidden_size=4096, intermediate_size=14336,
            num_layers=32, num_attention_heads=32, num_key_value_heads=8
        )
    }
    if name not in presets:
        raise ValueError(f"Preset {name} not found.")
    return presets[name]
