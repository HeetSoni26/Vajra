import json
from pathlib import Path


class ValidationEngine:
    """
    Validates models, checkpoints, configurations, and shard structures.
    """

    @staticmethod
    def validate_checkpoint(checkpoint_dir: str | Path) -> bool:
        checkpoint_dir = Path(checkpoint_dir)
        required_files = [
            "config.json",
        ]

        for f in required_files:
            if not (checkpoint_dir / f).exists():
                raise FileNotFoundError(f"Missing required checkpoint file: {f}")

        # Support both formats
        if (
            not (checkpoint_dir / "pytorch_model.bin").exists()
            and not (checkpoint_dir / "model.safetensors").exists()
        ):
            raise FileNotFoundError("Missing model weights (.bin or .safetensors)")

        return True

    @staticmethod
    def validate_configuration_compatibility(config1_path: Path, config2_path: Path) -> bool:
        with open(config1_path, "r") as f1, open(config2_path, "r") as f2:
            cfg1 = json.load(f1)
            cfg2 = json.load(f2)

        mismatches = []
        for key in ["vocab_size", "hidden_size", "num_layers", "num_attention_heads"]:
            if cfg1.get(key) != cfg2.get(key):
                mismatches.append(f"{key}: {cfg1.get(key)} != {cfg2.get(key)}")

        if mismatches:
            raise ValueError(f"Configuration mismatch: {', '.join(mismatches)}")

        return True
