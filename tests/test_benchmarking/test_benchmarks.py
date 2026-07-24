import json
from pathlib import Path
from unittest import mock


from benchmarks.benchmark import save_reports
from benchmarks.compare_benchmarks import generate_comparison_report
from benchmarks.runners.quality import compute_diversity_metrics


def test_compute_diversity_metrics():
    # Test identical tokens (0 diversity)
    text = "hello hello hello hello"
    metrics = compute_diversity_metrics(text)
    assert metrics["distinct_1"] == 0.25  # 1 unique / 4 total
    assert metrics["distinct_2"] == 0.3333  # 1 unique bigram / 3 total bigrams
    assert metrics["repetition_rate"] == 0.75

    # Test completely unique tokens
    text = "the quick brown fox"
    metrics = compute_diversity_metrics(text)
    assert metrics["distinct_1"] == 1.0
    assert metrics["distinct_2"] == 1.0
    assert metrics["repetition_rate"] == 0.0


def test_save_reports_and_comparison(tmp_path: Path):
    reports_dir = tmp_path / "reports"
    ckpt_1_dir = reports_dir / "checkpoint_100"
    ckpt_2_dir = reports_dir / "checkpoint_200"

    # Save mock reports
    save_reports(ckpt_1_dir, "checkpoint_100", {
        "checkpoint": "checkpoint_100",
        "validation_loss": 3.5,
        "perplexity": 33.1,
        "distinct_1": 0.8,
        "tokens_per_sec": 120.5,
        "inference_latency_first_token_ms": 15.2,
        "memory_ram_mb": 500
    })

    save_reports(ckpt_2_dir, "checkpoint_200", {
        "checkpoint": "checkpoint_200",
        "validation_loss": 3.0,
        "perplexity": 20.0,
        "distinct_1": 0.85,
        "tokens_per_sec": 125.0,
        "inference_latency_first_token_ms": 14.8,
        "memory_ram_mb": 510
    })

    # Verify JSON, CSV, MD generated
    assert (ckpt_1_dir / "benchmark.json").exists()
    assert (ckpt_1_dir / "benchmark.csv").exists()
    assert (ckpt_1_dir / "benchmark.md").exists()

    # Generate comparison
    generate_comparison_report(reports_dir)

    # Verify comparisons
    comp_json = reports_dir / "comparison.json"
    comp_csv = reports_dir / "comparison.csv"
    comp_md = reports_dir / "comparison.md"

    assert comp_json.exists()
    assert comp_csv.exists()
    assert comp_md.exists()

    # Verify data in JSON is sorted by step
    data = json.loads(comp_json.read_text())
    assert len(data) == 2
    assert data[0]["checkpoint"] == "checkpoint_100"
    assert data[1]["checkpoint"] == "checkpoint_200"


@mock.patch("benchmarks.benchmark.run_quality_benchmark")
@mock.patch("benchmarks.benchmark.run_performance_benchmark")
def test_benchmark_script_dispatch(mock_perf, mock_quality, tmp_path):
    from benchmarks.benchmark import main
    import sys

    ckpt_path = tmp_path / "checkpoint_step_50.pt"
    ckpt_path.touch()

    config_path = tmp_path / "config.yaml"
    config_path.touch()

    mock_quality.return_value = {"val_loss": 4.0}
    mock_perf.return_value = {"tokens_per_sec": 100}

    test_args = [
        "benchmark.py",
        "--checkpoint", str(ckpt_path),
        "--config", str(config_path),
        "--eval-dir", str(tmp_path)
    ]

    with mock.patch.object(sys, "argv", test_args):
        # Prevent it from actually saving to project root benchmarks/reports
        with mock.patch("benchmarks.benchmark.save_reports") as mock_save:
            main()

            mock_quality.assert_called_once()
            mock_perf.assert_called_once()
            mock_save.assert_called_once()
