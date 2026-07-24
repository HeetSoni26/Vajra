# Vajra Release & Model Packaging Framework

The **Vajra Release & Model Packaging Subsystem** provides an architecture-agnostic pipeline to package, verify, document, and publish any trained Vajra model checkpoint (`Vajra-57M`, `Vajra-125M`, `Vajra-370M`, `Vajra-1B`, `Vajra-2B`, etc.) into a fully reproducible release artifact ready for distribution on Hugging Face, GitHub Releases, or local inference deployments.

---

## 1. Packaging Workflow

The packaging process turns raw PyTorch `.pt` checkpoints, model configurations, and evaluation telemetry into a standardized Hugging Face compatible release directory.

### Command Line Usage
```bash
python -m release.package_model \
    --checkpoint checkpoints/pretrain_tiny_20260724_205603/checkpoint_step_250.pt \
    --config configs/training/pretrain_tiny.yaml \
    --model-name vajra-57m \
    --output-dir release/vajra-57m
```

### Python API Usage
```python
from release.package_model import package_model

package_dir = package_model(
    checkpoint_path="checkpoints/pretrain_tiny_20260724_205603/checkpoint_step_250.pt",
    config_path="configs/training/pretrain_tiny.yaml",
    model_name="vajra-57m",
    output_dir="release/vajra-57m"
)
```

---

## 2. Release Directory Layout

A completed release package strictly adheres to the following structure:

```
release/
└── vajra-57m/
    ├── model.safetensors          # Weights in SafeTensors format (or pytorch_model.bin)
    ├── tokenizer.json             # BPE Tokenizer vocabulary and merge rules
    ├── tokenizer_config.json      # Tokenizer parameters & special token mappings
    ├── special_tokens_map.json    # Special token definitions (<bos>, <eos>, <pad>, etc.)
    ├── config.json                # Model architecture & hyperparameter configuration
    ├── generation_config.json     # Default decoding parameters (temperature, top-p, etc.)
    ├── metadata.json              # Model telemetry & git commit information
    ├── manifest.json              # Complete reproducibility manifest with SHA-256 hashes
    ├── evaluation.json            # Validation loss and perplexity benchmarks
    ├── benchmark.json             # Hardware latency, throughput, and generation quality
    ├── README.md                  # Hugging Face Model Card with YAML metadata & citations
    ├── training_summary.md        # Executive markdown training report
    ├── training_summary.json      # Machine-readable training statistics
    ├── training_summary.csv       # Flattened tabular training metrics
    ├── verification_report.json   # Package integrity audit report
    └── checksums.txt              # Standard SHA-256 checksum index
```

---

## 3. Package Verification Workflow

Before a model is uploaded to Hugging Face or distributed, it must pass automated integrity verification.

The verification tool checks:
1. **Required Files**: Confirms all 15 release artifacts exist.
2. **Checksum Integrity**: Re-computes SHA-256 hashes for every file and compares against `checksums.txt`.
3. **Weight Loading**: Tests deserialization of state dictionaries (`model.safetensors` or `pytorch_model.bin`).
4. **Config & Tokenizer Parsing**: Validates JSON formatting and required architecture fields.
5. **Manifest & Metadata Consistency**: Confirms parameters, global step, and git hashes match across all reports.

### Running Verification
```bash
python -m release.verify_package --package-dir release/vajra-57m
```

---

## 4. Reproducibility Workflow

To guarantee reproducible releases, `package_model.py` automatically generates a comprehensive `manifest.json`. 

The manifest captures:
- Model Name, Architecture, and Exact Parameter Count
- Git Commit Hash at time of release packaging
- Complete Training & Model Hyperparameters
- Pretraining Dataset Name, Version, and Dataset Hash
- Exact Checkpoint Step and Total Tokens Processed
- Timestamps for Evaluation, Benchmarking, and Packaging

---

## 5. Model Card & Reports Generation

- **`create_model_card.py`**: Generates a standard Hugging Face `README.md` complete with frontmatter tags, architecture breakdown, evaluation metrics table, benchmark performance telemetry, sample generations, intended usage guidelines, and BibTeX citation.
- **`create_training_report.py`**: Generates structured `training_summary.json`, `training_summary.csv`, and `training_summary.md` detailing optimizer settings, loss trajectories, and throughput statistics.
