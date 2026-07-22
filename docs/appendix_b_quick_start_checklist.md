# Appendix B — Quick-Start Checklist

Use this checklist to verify Phase 0 and early implementation readiness.

## Repository and environment

- [ ] repository structure created
- [ ] `environment.yml` exists
- [ ] `pyproject.toml` exists
- [ ] `Dockerfile` builds
- [ ] `.gitignore` excludes secrets, data, and checkpoints
- [ ] tests run locally
- [ ] CI workflow exists
- [ ] pre-commit hooks configured

## GPU and training stack

- [ ] cloud GPU access confirmed
- [ ] `torch.cuda.is_available()` is true on target host
- [ ] CUDA, driver, NCCL, and PyTorch versions recorded
- [ ] FlashAttention or chosen attention backend tested
- [ ] DeepSpeed debug launch tested
- [ ] W&B or local metric logging tested

## Tokenizer

- [ ] tokenizer corpus sources selected
- [ ] license/provenance recorded
- [ ] tokenizer trained
- [ ] round-trip tests pass
- [ ] compression and fertility report complete
- [ ] tokenizer version frozen

## Dataset

- [ ] source/license audit complete
- [ ] download scripts tested
- [ ] processing pipeline stages implemented
- [ ] deduplication tested
- [ ] contamination checks implemented
- [ ] tokenized shards created
- [ ] dataset manifest generated

## Model and training

- [ ] parameter count verified
- [ ] forward pass tests pass
- [ ] gradient flow verified
- [ ] checkpoint save/load tested
- [ ] 125M debug run completes
- [ ] debug validation perplexity improves

## Release preparation

- [ ] model card draft exists
- [ ] technical report template exists
- [ ] benchmark config exists
- [ ] deployment stubs exist
- [ ] risk register reviewed
