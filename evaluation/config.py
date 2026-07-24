from pathlib import Path

from pydantic import BaseModel


class EvaluationConfig(BaseModel):
    batch_size: int = 16
    max_sequence_length: int = 2048
    mixed_precision: str = "bf16"  # none, fp16, bf16
    device: str = "cuda"

    validation_datasets: list[str] = ["output/shards"]
    metrics: list[str] = ["loss", "perplexity", "accuracy", "bpb", "throughput"]
    benchmarks: list[str] = []  # e.g. "hellaswag", "arc"

    output_dir: str = "output/evaluations"
    evaluation_interval: int = 1000  # for potential integration later

    def save(self, path: Path | str):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            f.write(self.model_dump_json(indent=2))

    @classmethod
    def load(cls, path: Path | str) -> "EvaluationConfig":
        with open(path, "r") as f:
            return cls.model_validate_json(f.read())
