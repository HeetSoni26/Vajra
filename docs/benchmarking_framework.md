# Vajra Benchmarking Framework

The Vajra Benchmarking Framework provides an automated suite to measure the objective quality and performance characteristics of model checkpoints.

## Architecture

The framework consists of:
- **`benchmarks/benchmark.py`**: The primary runner that orchestrates quality and performance suites.
- **`benchmarks/compare_benchmarks.py`**: Aggregates all benchmark reports into unified leaderboards.
- **`benchmarks/runners/quality.py`**: Collects validation metrics (loss, perplexity) and computes text generation quality metrics (Distinct-1, Distinct-2, Repetition Rate).
- **`benchmarks/runners/performance.py`**: Measures inference latency, throughput (tokens/sec), loading times, and memory usage natively.

## Workflow

The benchmarking suite can be executed independently:
```bash
python -m benchmarks.benchmark \
    --checkpoint checkpoints/run_1/checkpoint_step_100.pt \
    --config configs/training/pretrain_tiny.yaml \
    --eval-dir evaluations
```

Or fully integrated into the automated evaluation pipeline:
```bash
python -m evaluation.evaluate_all \
    --experiment-dir checkpoints/run_1 \
    --config configs/training/pretrain_tiny.yaml \
    --run-benchmarks
```

## Report Formats

When benchmarking completes, reports are generated inside `benchmarks/reports/checkpoint_XYZ/`:
- `benchmark.json`: Raw telemetry data.
- `benchmark.csv`: Formatted key-value pairs for easy ingestion.
- `benchmark.md`: A readable markdown summary including a generated sample.

A consolidated set of comparison files is maintained at `benchmarks/reports/`:
- `comparison.json`, `comparison.csv`, `comparison.md`

These comparisons help track model improvements across checkpoints directly correlating training progression with inference efficiency and generation quality.
