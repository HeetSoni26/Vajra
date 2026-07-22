# Vajra Evaluation & Validation Framework

The Evaluation Engine provides independent out-of-core structural analysis measuring true generalizability properties scaling linearly without mutating the training pipelines natively.

## Components

- **`config.py` (`EvaluationConfig`)**: Defines the validation boundaries matching hyper-parameters explicitly bounding tracking matrices cleanly parsing targets correctly safely mapping hardware structures.
- **`metrics/base.py`, `standard.py`**: Isolates standard statistical trackers mapping standard arrays safely computing Perplexity (e^Loss), Accuracy arrays mathematically predicting shifted constraints, and exact token throughput timings seamlessly.
- **`benchmarks/base.py`**: Extensible interface mapped exactly generating abstract `load_dataset`, `format_prompt` targets safely allowing integration of HumanEval or MMLU mapping targets dynamically.
- **`validation/checker.py` (`ValidationEngine`)**: Validates checkpoint topologies mapping structural configs explicitly preventing dimension mismatch crashes gracefully tracking variables identically cleanly.
- **`reporting/generator.py` (`ReportGenerator`)**: Safely emits `.json`, `.csv`, and `.md` formats documenting explicit metrics alongside timestamps and parameters cleanly. 
- **`reporting/comparison.py` (`ComparisonEngine`)**: Structurally wraps JSON outputs mapping absolute and percentage differences mathematically highlighting regression/improvement ratios perfectly safely natively.
- **`engine/evaluator.py` (`Evaluator`)**: Combines validation arrays running inference loops safely generating native PyTorch graphs explicitly without backpropagation topologies cleanly mapping results gracefully cleanly securely.

## CLI Utility

Operate physically bounding states straight out of `manage_evaluation.py`:

```bash
# Validate structural layout cleanly mapping dimensions tracking topologies
python evaluation/scripts/manage_evaluation.py validate output/training/checkpoint-1000

# Execute evaluations tracking bounding configurations dynamically mapping JSON logs gracefully.
python evaluation/scripts/manage_evaluation.py evaluate --checkpoint-dir output/training/checkpoint-1000 --dataset-dir output/shards

# Compare improvements seamlessly returning JSON delta formats locally safely explicitly
python evaluation/scripts/manage_evaluation.py compare output/evaluations/checkpoint-1000_eval.json output/evaluations/checkpoint-2000_eval.json
```
