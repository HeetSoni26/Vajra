# Section 11 — Risk Register

## 11.1 Objective

Track major technical, operational, legal, financial, and project-execution risks. Each risk has likelihood, impact, detection signal, and mitigation.

## 11.2 Risk table

| # | Risk | Likelihood | Impact | Detection signal | Mitigation |
|---:|---|---|---|---|---|
| 1 | Training divergence | Medium | High | loss/grad norm spike, NaNs | BF16, warmup, grad clipping, checkpoint rollback, LR reduction |
| 2 | Low-quality data | Medium | High | poor generations, high perplexity | filtering, dedup, manual samples, debug-model comparisons |
| 3 | Compute cost overrun | Medium | High | cost/token exceeds budget | budget alerts, spot/on-demand mix, debug runs, early stop gates |
| 4 | GPU unavailability | Medium | Medium | interrupted/preempted jobs | multi-provider accounts, frequent checkpoints |
| 5 | Benchmark contamination | Low | High | suspicious benchmark gains | n-gram overlap checks, source provenance, held-out eval |
| 6 | Training OOM | Medium | Medium | CUDA OOM | reduce microbatch, checkpoint activations, ZeRO-3 fallback |
| 7 | Tokenizer quality failure | Low | Medium | poor code/math fertility | evaluate before pretraining, retrain tokenizer early |
| 8 | Silent data corruption | Medium | High | loss spikes tied to shards | shard checksums, bad-batch fingerprints, validation jobs |
| 9 | SFT catastrophic forgetting | Medium | Medium | base benchmarks regress | fewer epochs, lower LR, mixed capability data |
| 10 | Dataset license issue | Low | High | license audit gap | use permissive sources, record provenance, remove disputed data |
| 11 | Slow preprocessing | Medium | Medium | pipeline throughput too low | streaming, sharding, parallel workers, profile bottlenecks |
| 12 | Inference incompatibility | Low | Medium | HF/GGUF/vLLM load failures | test conversion on debug checkpoint first |
| 13 | Evaluation harness error | Low | High | inconsistent baseline reproduction | pin versions, cross-check known public models |
| 14 | Security issue in API | Medium | High | unsafe input handling, no limits | request validation, max token caps, rate limits, auth hooks |
| 15 | Secrets leak | Low | High | committed `.env`/tokens | `.gitignore`, pre-commit secret scan, environment-only credentials |
| 16 | Researcher burnout | Medium | High | stalled milestones | weekly logs, smaller gates, publish intermediate artifacts |
| 17 | Architecture underperforms | Low | High | debug ablations lag baselines | run 125M ablations before full-scale training |
| 18 | Quantization quality loss | Medium | Medium | PPL jump after Q4 | test Q5/Q8, calibrate quantization, document tradeoffs |
| 19 | Alignment degrades truthfulness | Medium | Medium | TruthfulQA decline | preference-data audit, reduce DPO strength, add eval gates |
| 20 | Public misuse concerns | Medium | Medium | unsafe generations | model card limitations, safety eval, release usage guidance |

## 11.3 Risk handling process

1. Assign each risk an owner before the corresponding phase begins.
2. Define monitoring signals in the run dashboard or release checklist.
3. Review active risks at every milestone checkpoint.
4. Record incidents in `docs/notes/` with root cause and corrective action.
5. Do not proceed to public release if high-impact legal, safety, or reproducibility risks remain unresolved.

## 11.4 Training incident protocol

For divergence, NaNs, or repeated loss spikes:

1. stop the run or pause checkpoint promotion
2. preserve logs and bad-batch fingerprints
3. restore from the last stable checkpoint
4. inspect data shard and recent code/config changes
5. rerun a short validation job
6. document the incident before resuming full training

## 11.5 Legal and data-risk protocol

If any dataset source has unclear licensing:

1. quarantine the source
2. remove it from downstream tokenized shards
3. regenerate affected manifests
4. document removal and replacement source
5. rerun contamination and quality checks

## 11.6 Release-blocking risks

The following block public release:

- unresolved dataset license uncertainty
- missing model card limitations
- unreproducible benchmark results
- known severe API security issue
- model checkpoint cannot load through documented paths
- benchmark contamination not checked
- no rollback/reproducibility record for final checkpoint
