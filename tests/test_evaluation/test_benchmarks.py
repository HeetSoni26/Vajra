from evaluation.benchmarks.registry import BenchmarkRegistry
from evaluation.benchmarks.pipeline import EvaluationPipeline
from evaluation.benchmarks.integration import TrainingBenchmarkIntegration
# Load adapters


def test_benchmark_registry():
    benchmarks = BenchmarkRegistry.list_benchmarks()
    assert "hellaswag" in benchmarks
    assert "mmlu" in benchmarks
    assert "gsm8k" in benchmarks


def test_evaluation_pipeline():
    pipeline = EvaluationPipeline(model=None)
    mock_data = {
        "hellaswag": [{"ctx": "Hello", "label": "world"}],
        "mmlu": [{"question": "Q", "label": "A"}],
    }
    results = pipeline.run_suite(["hellaswag", "mmlu"], mock_data)

    assert "hellaswag" in results
    assert "mmlu" in results
    assert "accuracy" in results["hellaswag"]
    assert "latency" in results["hellaswag"]


def test_benchmark_integration(tmp_path):
    output_dir = tmp_path / "benchmarks"
    integration = TrainingBenchmarkIntegration(
        model=None, output_dir=str(output_dir), benchmarks=["hellaswag"]
    )

    results = integration.evaluate_checkpoint("ckpt-100")

    assert "hellaswag" in results
    assert (output_dir / "ckpt-100_results.json").exists()
    assert (output_dir / "ckpt-100_results.csv").exists()
    assert (output_dir / "ckpt-100_report.md").exists()
