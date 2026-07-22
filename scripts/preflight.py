import sys
import shutil
import torch
from pathlib import Path

def run_preflight_checks(output_dir: str, dataset_dir: str):
    """Verifies all components are ready for training."""
    print("Running Vajra-370M Pre-flight Validation...")
    errors = []

    # 1. Tokenizer
    if not Path("tokenizer").exists():
        errors.append("Tokenizer directory is missing.")
        
    # 2. Dataset / Binary Shards
    if not Path(dataset_dir).exists():
        errors.append(f"Dataset directory '{dataset_dir}' is missing.")
    else:
        # Check for .bin files
        shards = list(Path(dataset_dir).glob("*.bin"))
        if not shards:
            errors.append(f"No binary shards (.bin) found in {dataset_dir}.")

    # 3. Model Architecture / Optimizer
    try:
        from model.modeling import VajraForCausalLM  # noqa: F401
        from training.workflows.preset import get_vajra_370m_preset  # noqa: F401
    except ImportError as e:
        errors.append(f"Failed to import core modules: {e}")

    # 4. Checkpoint Directory
    out = Path(output_dir)
    try:
        out.mkdir(parents=True, exist_ok=True)
    except Exception as e:
        errors.append(f"Cannot write to output directory {output_dir}: {e}")

    # 5. Disk Space
    total, used, free = shutil.disk_usage(str(out.resolve()))
    free_gb = free // (2**30)
    if free_gb < 50:
        errors.append(f"Insufficient disk space. Only {free_gb}GB available. 50GB required.")

    # 6. GPU Memory / Availability
    if not torch.cuda.is_available():
        errors.append("CUDA is not available. GPU is required for production training.")
    else:
        for i in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(i)
            mem_gb = props.total_memory // (2**30)
            if mem_gb < 8:
                errors.append(f"GPU {i} ({props.name}) has only {mem_gb}GB VRAM. 8GB minimum required.")

    if errors:
        print("\nPre-flight Checks FAILED!")
        for err in errors:
            print(f" - {err}")
        sys.exit(1)
    else:
        print("\nAll systems GO! Pre-flight checks passed successfully.")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", default="checkpoints")
    parser.add_argument("--dataset-dir", default="data/tokenized")
    args = parser.parse_args()
    
    run_preflight_checks(args.output_dir, args.dataset_dir)
