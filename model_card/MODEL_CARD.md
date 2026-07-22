# Model Card: Vajra LM

## Model description

Foundation LM is a planned 1B–2B parameter decoder-only language model trained from random initialization. This scaffold defines the engineering path, but does not include trained weights.

## Intended use

Research, fine-tuning experiments, local inference after training, and educational reproduction of small foundation-model training pipelines.

## Limitations

No trained checkpoint is included in this package. Dataset licensing, safety evaluation, benchmark results, and deployment claims must be completed after training.

## Training data

The intended mix is English web, code, mathematics, scientific text, books, Wikipedia, technical Q&A, and a small multilingual component. Final source manifests and licenses must be documented before release.

## Evaluation

Use `evaluation/run_lm_eval.py` and `evaluation/run_codegen_eval.py` after checkpoints are available.

## License

The scaffold is prepared for Apache-2.0 release. Confirm all downstream artifacts are compatible before public distribution.
