from pathlib import Path
from typing import Any

from release.export import ModelExporter
from release.model_card import ModelCardGenerator
from release.verify_package import verify_package as full_verify_package


class ReleasePackager:
    """Packages the final release artifacts into a distributable directory."""

    def __init__(self, output_dir: str | Path):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def create_package(
        self,
        model: Any,
        config: dict[str, Any],
        tokenizer: Any = None,
        metrics: dict[str, Any] = None,
    ):
        """Creates a complete release package."""
        exporter = ModelExporter(self.output_dir)
        exporter.export_huggingface(model, config, tokenizer)

        card_gen = ModelCardGenerator(self.output_dir)
        card_gen.generate(config.get("model_name", "Vajra-57M"), config, metrics)

        root_license = Path("LICENSE")
        if root_license.exists():
            (self.output_dir / "LICENSE").write_text(root_license.read_text(encoding="utf-8"), encoding="utf-8")
        else:
            (self.output_dir / "LICENSE").write_text("MIT License / Apache 2.0 License\n", encoding="utf-8")

        return self.output_dir

    def verify_package(self) -> bool:
        """Validates that all essential components of the release are present and valid."""
        passed, _ = full_verify_package(self.output_dir)
        return passed
