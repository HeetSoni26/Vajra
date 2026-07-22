# Section 14 — Timeline

## 14.1 Objective

Provide a dependency-aware execution sequence for the project. The timeline is a planning artifact, not a guarantee. Actual duration depends on compute availability, data throughput, debugging complexity, and release-review requirements.

## 14.2 Phase sequence

```text
Phase 0  Vajra and environment
Phase 1  Tokenizer
Phase 2  Dataset pipeline
Phase 3  Model implementation and debug training
Phase 4  Pretraining
Phase 5  Instruction tuning and alignment
Phase 6  Evaluation, quantization, and optimization
Phase 7  Deployment and public release
```

## 14.3 Dependency graph

```text
Phase 0
  └── Phase 1 tokenizer
        └── Phase 2 dataset tokenization
              └── Phase 3 debug training
                    └── Phase 4 pretraining
                          └── Phase 5 SFT/DPO
                                └── Phase 6 final eval/quantization
                                      └── Phase 7 release
```

Documentation, API scaffolding, SDK development, and release templates can be prepared while data processing or pretraining runs, but final claims must wait for measured results.

## 14.4 Work breakdown by milestone

| Milestone | Main outputs | Gate |
|---|---|---|
| Vajra ready | environment, CI, repo, docs skeleton | tests and GPU smoke pass |
| Tokenizer frozen | tokenizer artifacts and report | compression/fertility targets met |
| Dataset v1 ready | tokenized shards and manifest | source/license/filter stats complete |
| Debug model ready | 125M run and restore test | loss decreases and checkpoint restores |
| Base checkpoint ready | final pretrained model | validation and milestone benchmarks complete |
| Instruct checkpoint ready | SFT model | instruction adherence and no severe regressions |
| Aligned checkpoint ready | DPO model | preference eval and safety checks complete |
| Release candidate ready | model card, docs, quantizations, API | all release gates pass |

## 14.5 Parallelizable work

Can run in parallel with data processing or pretraining:

- writing architecture, tokenizer, and dataset docs
- building evaluation scripts
- implementing API and SDK stubs
- preparing HuggingFace model card
- preparing technical report template
- testing GGUF conversion on debug checkpoints
- constructing qualitative prompt suites

Should not be parallelized before prerequisites:

- final tokenizer-dependent tokenization before tokenizer is frozen
- main pretraining before dataset validation
- SFT before base checkpoint validation
- public release before final evaluation and model-card completion

## 14.6 Compute-constrained path

If compute is limited:

1. train the debug model first
2. train the 1B model before the 2B model
3. reduce token budget only after documenting the expected quality tradeoff
4. prioritize data quality over parameter count
5. publish transparent intermediate results rather than overstating final quality

## 14.7 Project tracking artifacts

Maintain:

- `docs/release_checklist.md`
- `docs/phase_gate_checklist.md`
- `docs/notes/` experiment logs
- dataset manifests
- benchmark result tables
- budget/cost log
- issue tracker labels for each phase

## 14.8 Release readiness gate

The project should move to public release only when:

- all critical documentation is complete
- all benchmark results are reproducible
- licensing review is complete
- model loads in documented formats
- safety and limitation notes are included
- final artifacts match the model card
