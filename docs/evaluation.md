# Model Evaluation Framework

The `ModelEvaluator` module (`evaluation/evaluator.py`) measures language modeling performance on binary `uint32` token datasets (`train.bin`, `val.bin`, `test.bin`).

## Key Metrics Computed

- **Cross-Entropy Loss**: Average log-likelihood loss per token.
- **Perplexity (PPL)**: $e^{\text{CrossEntropy}}$. Measures model uncertainty. Lower is better.
- **Bits Per Character (BPC)**: Loss divided by $\ln(2)$.
- **Tokens/sec Evaluation Speed**: Ingestion throughput during evaluation.

## Execution

```bash
# Evaluate on all dataset splits
python evaluation/evaluator.py \
  --config configs/training/pretrain_tiny.yaml \
  --checkpoint checkpoints/run/latest.pt \
  --data_dir data/tokenized \
  --output evaluation_report.json
```

## Output Structure (`evaluation_report.json`)

```json
{
  "splits": {
    "val.bin": {
      "avg_cross_entropy": 2.8468,
      "perplexity": 17.23,
      "bits_per_character": 4.107,
      "tokens_per_sec": 30877.0,
      "total_tokens": 6400
    }
  },
  "model_info": {
    "total_parameters": 825600,
    "hidden_size": 64,
    "num_layers": 2
  }
}
```
