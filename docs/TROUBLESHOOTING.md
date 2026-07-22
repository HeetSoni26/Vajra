# Vajra Troubleshooting Guide

### 1. Out of Memory (OOM) Errors
- **Symptom**: `CUDA out of memory`
- **Fix**: Reduce `batch_size_per_gpu` in `scripts/recipes.py`. Enable `gradient_checkpointing`. Ensure `precision=bf16`.

### 2. Loss Spikes / NaNs
- **Symptom**: Loss suddenly jumps or hits NaN.
- **Fix**: Ensure `NumericalStabilityWatchdog` is enabled in `ProductionConfig`. It will automatically skip the step. If persistent, reduce `learning_rate` or adjust `gradient_clipping`.

### 3. DDP Hangs
- **Symptom**: Training stalls at step initialization on multiple GPUs.
- **Fix**: Ensure `os.environ["USE_LIBUV"] = "0"` is set on Windows. Ensure NCCL backend is correctly compiled (Linux). Use `Gloo` backend for debugging.
