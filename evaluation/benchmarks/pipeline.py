import time
from typing import List, Dict, Any
from evaluation.benchmarks.registry import BenchmarkRegistry


class EvaluationPipeline:
    """Executes benchmark evaluations and collects metrics."""

    def __init__(self, model: Any, tokenizer: Any = None):
        self.model = model
        self.tokenizer = tokenizer

    def evaluate_benchmark(
        self, benchmark_name: str, dataset: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Runs evaluation for a single benchmark."""
        adapter = BenchmarkRegistry.get_adapter(benchmark_name)

        predictions = []
        references = []

        start_time = time.time()

        for item in dataset:
            # Mock execution
            adapter.format_prompt(item)
            predictions.append("mock_pred")
            references.append(item.get("label", "mock_pred"))

        latency = time.time() - start_time
        tokens_per_sec = (len(dataset) * 10) / latency if latency > 0 else 0

        metrics = adapter.compute_metrics(predictions, references)
        metrics.update(
            {
                "latency": latency,
                "tokens_per_sec": tokens_per_sec,
                "perplexity": 0.0,
                "loss": 0.0,
                "memory_usage_mb": 0.0,
            }
        )

        return metrics

    def run_suite(
        self, benchmarks: List[str], dataset_mocks: Dict[str, List[Dict[str, Any]]]
    ) -> Dict[str, Dict[str, Any]]:
        """Runs evaluation across multiple benchmarks."""
        results = {}
        # In a real environment, this might use concurrent.futures for parallel execution.
        for bench in benchmarks:
            data = dataset_mocks.get(bench, [])
            results[bench] = self.evaluate_benchmark(bench, data)

        return results
