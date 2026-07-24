from pathlib import Path
from typing import Dict, Any

from release.export import ModelExporter
from release.model_card import ModelCardGenerator


class ReleasePackager:
    """Packages the final release artifacts into a distributable directory."""

    def __init__(self, output_dir: str):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def create_package(
        self,
        model: Any,
        config: Dict[str, Any],
        tokenizer: Any = None,
        metrics: Dict[str, Any] = None,
    ):
        """Creates a complete release package."""

        # 1. HuggingFace Export (Weights, config.json, generation_config.json, tokenizer)
        exporter = ModelExporter(self.output_dir)
        exporter.export_huggingface(model, config, tokenizer)

        # 2. Model Card
        card_gen = ModelCardGenerator(self.output_dir)
        card_gen.generate("Vajra-370M", config, metrics)

        # 3. License & Changelog Placeholders
        (self.output_dir / "LICENSE").write_text("MIT License")
        (self.output_dir / "CHANGELOG.md").write_text(
            "# Changelog\n\n## 1.0.0\n- Initial Release of Vajra-370M"
        )

        return self.output_dir

    def verify_package(self) -> bool:
        """Validates that all essential components of the release are present."""
        required_files = ["config.json", "generation_config.json", "README.md", "LICENSE"]

        for file in required_files:
            if not (self.output_dir / file).exists():
                return False

        # Check for weights (either safetensors or bin)
        has_weights = (self.output_dir / "model.safetensors").exists() or (
            self.output_dir / "pytorch_model.bin"
        ).exists()

        return has_weights
