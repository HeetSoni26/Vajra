from typing import Any

from evaluation.benchmarks.pipeline import EvaluationPipeline
from evaluation.benchmarks.reporting import BenchmarkReporter


class TrainingBenchmarkIntegration:
    """Hooks benchmarks into the training orchestration."""

    def __init__(self, model: Any, output_dir: str, benchmarks: list[str]):
        self.pipeline = EvaluationPipeline(model)
        self.reporter = BenchmarkReporter(output_dir)
        self.benchmarks = benchmarks

    def evaluate_checkpoint(
        self, checkpoint_name: str, mock_data: dict[str, list[dict[str, Any]]] = None
    ) -> dict[str, Any]:
        """Evaluates a checkpoint automatically."""
        if mock_data is None:
            mock_data = {b: [{"question": "mock?", "label": "mock"}] for b in self.benchmarks}

        results = self.pipeline.run_suite(self.benchmarks, mock_data)

        self.reporter.generate_json(checkpoint_name, results)
        self.reporter.generate_csv(checkpoint_name, results)
        self.reporter.generate_markdown(checkpoint_name, results)

        return results
