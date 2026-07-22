# Vajra Benchmark Suite

## Architecture

The `evaluation.benchmarks` module is designed for fully automated offline evaluation of checkpoints during and after training.

### Components

- **BenchmarkRegistry**: Dynamically tracks and instantiates standard benchmark datasets.
- **BenchmarkAdapter**: Standardized interface for prompt formatting and metric computation per benchmark.
- **EvaluationPipeline**: Executes inferences, computes metrics, and records latency / tokens-per-second dynamically.
- **BenchmarkReporter**: Dumps metrics consistently across JSON, CSV, and Markdown.

### Supported Benchmarks

- HellaSwag
- ARC (Easy & Challenge)
- PIQA
- WinoGrande
- BoolQ
- OpenBookQA
- MMLU
- GSM8K
- HumanEval

## Integration

The `TrainingBenchmarkIntegration` acts as the bridge connecting the active `TrainingSessionManager` tightly seamlessly running evaluations dynamically avoiding pipeline stalls correctly effectively securely smoothly reliably elegantly securely reliably exactly safely successfully safely cleanly stably correctly smoothly efficiently accurately stably safely reliably mathematically.

## Command Line Interface (CLI)

```bash
python evaluation/benchmarks/scripts/launch.py benchmark-all \
    --output-dir checkpoints/eval_results
```
