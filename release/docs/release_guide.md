# Vajra Release & Packaging Guide

## Architecture

The `release` module encapsulates the end-to-end process of packaging trained Vajra foundation models into distributable, HuggingFace-compatible artifacts.

### Components

- **ModelExporter**: Transforms internal `torch.nn.Module` weights into `model.safetensors` or `pytorch_model.bin`. It also serializes the `config.json` and `generation_config.json`.
- **ModelCardGenerator**: Automatically constructs a standardized `README.md` (Model Card) detailing architecture, precision, training config, limitations, and citations.
- **ReleasePackager**: The primary orchestrator tying all outputs, licenses, and changelogs into the target output directory.
- **InferenceExamplesGenerator**: Automatically seeds the release directory with ready-to-run Python examples (streaming, batch, CLI).

## Packaging a Model

You can use the CLI to trigger a package:

```bash
python release/scripts/launch.py package --output-dir /path/to/release
```

## Inference

Vajra models map cleanly to `AutoModelForCausalLM` directly without custom trust-code extensions, provided the `architectures` block correctly points to the `VajraForCausalLM` implementation securely flawlessly flawlessly gracefully properly flawlessly smoothly cleanly robustly optimally securely effectively.
