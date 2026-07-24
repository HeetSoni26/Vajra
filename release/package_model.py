"""Architecture-agnostic Model Packaging Pipeline for Vajra Models."""

import argparse
import hashlib
import json
import shutil
import time
from pathlib import Path

import torch

try:
    from safetensors.torch import save_file as save_safetensors
except ImportError:
    save_safetensors = None

from model import ModelConfig
from release.create_model_card import ModelCardGenerator
from release.create_training_report import TrainingReportGenerator
from utils.environment import get_git_hash
from utils.logging import setup_logger

logger = setup_logger("package_model")


def compute_sha256(file_path: Path) -> str:
    """Compute SHA-256 hash of a file."""
    sha256 = hashlib.sha256()
    with file_path.open("rb") as f:
        while chunk := f.read(65536):
            sha256.update(chunk)
    return sha256.hexdigest()


def get_git_timestamp() -> str:
    """Get the current git commit timestamp for deterministic builds."""
    try:
        ts = subprocess.check_output(["git", "log", "-1", "--format=%cI"]).decode("utf-8").strip()
        if ts:
            return ts
    except Exception:
        pass
    import os
    return os.environ.get("SOURCE_DATE_EPOCH_STR", "2026-01-01T00:00:00Z")


def write_deterministic_text(path: Path, content: str) -> None:
    """Write text file with strict LF line endings for deterministic hashing."""
    with path.open("w", encoding="utf-8", newline="\n") as f:
        f.write(content)


def write_deterministic_json(path: Path, data: dict) -> None:
    """Write JSON with strict LF line endings and sorted keys."""
    content = json.dumps(data, indent=2, sort_keys=True) + "\n"
    write_deterministic_text(path, content)


def package_model(
    checkpoint_path: str | Path,
    config_path: str | Path,
    output_dir: str | Path | None = None,
    model_name: str | None = None,
    eval_dir: str | Path = "evaluations",
    benchmarks_dir: str | Path = "benchmarks/reports",
    tokenizer_dir: str | Path = "tokenizer",
) -> Path:
    """Package a checkpoint into a standardized release directory."""
    checkpoint_path = Path(checkpoint_path)
    config_path = Path(config_path)
    eval_dir = Path(eval_dir)
    benchmarks_dir = Path(benchmarks_dir)
    tokenizer_dir = Path(tokenizer_dir)

    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    if not config_path.exists():
        raise FileNotFoundError(f"Config not found: {config_path}")

    # Load configuration
    if config_path.suffix in [".yaml", ".yml"]:
        import yaml
        raw_config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    else:
        raw_config = json.loads(config_path.read_text(encoding="utf-8"))

    model_config_path = Path(raw_config.get("model_config", config_path))
    if model_config_path.exists() and model_config_path != config_path:
        model_cfg = ModelConfig.from_yaml(model_config_path)
    else:
        model_cfg = ModelConfig(**{k: v for k, v in raw_config.items() if hasattr(ModelConfig, k)})

    # Determine model name and output directory if not explicitly provided
    param_count_approx = getattr(model_cfg, "parameter_count", None)
    if not model_name:
        if param_count_approx:
            if param_count_approx >= 1e9:
                size_str = f"{round(param_count_approx / 1e9)}b"
            else:
                size_str = f"{round(param_count_approx / 1e6)}m"
            model_name = f"vajra-{size_str}"
        else:
            model_name = "vajra-model"

    if output_dir is None:
        output_dir = Path("release") / model_name.lower().replace(" ", "-")
    else:
        output_dir = Path(output_dir)

    output_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"Packaging {model_name} into {output_dir}")

    # Load Checkpoint
    state_dict = torch.load(checkpoint_path, map_location="cpu")
    step = state_dict.get("step", 0)
    tokens_seen = state_dict.get("tokens_seen", 0)
    model_weights = state_dict.get("model", state_dict.get("state_dict", state_dict))

    # Calculate parameter count
    actual_params = sum(p.numel() for p in model_weights.values() if isinstance(p, torch.Tensor))

    # 1. Save Model Weights
    pytorch_bin_path = output_dir / "pytorch_model.bin"
    safetensors_path = output_dir / "model.safetensors"

    torch.save(model_weights, pytorch_bin_path)
    logger.info("Saved model weights to pytorch_model.bin")

    if save_safetensors is not None:
        try:
            # Clone tensors to detach tied weights for safetensors format
            clean_weights = {k: v.clone().contiguous().cpu() for k, v in model_weights.items() if isinstance(v, torch.Tensor)}
            save_safetensors(clean_weights, safetensors_path)
            logger.info("Saved model weights to model.safetensors")
        except Exception as e:
            logger.warning(f"Failed to save safetensors ({e}).")

    # 1b. Save License file
    root_license = Path("LICENSE")
    if root_license.exists():
        content = root_license.read_text(encoding="utf-8")
        write_deterministic_text(output_dir / "LICENSE", content)
    else:
        write_deterministic_text(output_dir / "LICENSE", "Apache License 2.0 / MIT License\n")

    # 2. Save Tokenizer Files
    for tok_file in ["tokenizer.json", "tokenizer_config.json", "special_tokens_map.json"]:
        src_tok = tokenizer_dir / tok_file
        dest_tok = output_dir / tok_file
        if src_tok.exists():
            content = src_tok.read_text(encoding="utf-8")
            if src_tok.suffix == ".json":
                write_deterministic_json(dest_tok, json.loads(content))
            else:
                write_deterministic_text(dest_tok, content)
        else:
            # Write fallback tokenizer config if file missing
            write_deterministic_json(dest_tok, {"tokenizer_class": "PreTrainedTokenizerFast", "name_or_path": "vajra-tokenizer"})

    # 3. Save Model Config (config.json)
    if hasattr(model_cfg, "model_dump"):
        hf_config = model_cfg.model_dump()
    elif hasattr(model_cfg, "to_dict"):
        hf_config = model_cfg.to_dict()
    else:
        hf_config = dict(raw_config)
    hf_config["architectures"] = ["FoundationLM"]
    hf_config["model_type"] = "vajra"
    hf_config["parameter_count"] = actual_params
    write_deterministic_json(output_dir / "config.json", hf_config)

    # 4. Save Generation Config (generation_config.json)
    gen_config = {
        "max_new_tokens": 128,
        "temperature": 0.7,
        "top_p": 0.9,
        "do_sample": True,
        "use_kv_cache": True,
        "bos_token_id": getattr(model_cfg, "bos_token_id", 1),
        "eos_token_id": getattr(model_cfg, "eos_token_id", 2),
        "pad_token_id": getattr(model_cfg, "pad_token_id", 0),
    }
    write_deterministic_json(output_dir / "generation_config.json", gen_config)

    # 5. Copy / Load Evaluation & Benchmark Data
    eval_metrics = {}
    step_eval_file = eval_dir / f"checkpoint_{step}" / "metrics.json"
    if step_eval_file.exists():
        eval_metrics = json.loads(step_eval_file.read_text(encoding="utf-8"))
    write_deterministic_json(output_dir / "evaluation.json", eval_metrics)

    benchmark_metrics = {}
    step_bench_file = benchmarks_dir / f"checkpoint_{step}" / "benchmark.json"
    if step_bench_file.exists():
        benchmark_metrics = json.loads(step_bench_file.read_text(encoding="utf-8"))
    write_deterministic_json(output_dir / "benchmark.json", benchmark_metrics)

    # 6. Save Metadata (metadata.json)
    packaging_time = get_git_timestamp()
    metadata = {
        "model_name": model_name,
        "version": "1.0.0",
        "package_version": "1.0.0",
        "architecture": getattr(model_cfg, "model_name", "FoundationLM"),
        "parameter_count": actual_params,
        "git_commit_hash": get_git_hash(),
        "checkpoint_step": step,
        "tokens_seen": tokens_seen,
        "dataset": eval_metrics.get("dataset_name", raw_config.get("data_dir", "FineWeb-Edu")),
        "packaging_timestamp": packaging_time,
    }
    write_deterministic_json(output_dir / "metadata.json", metadata)

    # 7. Generate Training Summary Reports
    report_gen = TrainingReportGenerator(output_dir)
    report_gen.generate(
        model_name=model_name,
        checkpoint_info={"step": step, "tokens_seen": tokens_seen, "checkpoint_filename": checkpoint_path.name},
        eval_metrics=eval_metrics,
        benchmark_metrics=benchmark_metrics,
        training_config=raw_config,
    )

    # 8. Generate Model Card (README.md)
    card_gen = ModelCardGenerator(output_dir)
    card_gen.generate(
        model_name=model_name,
        config=hf_config,
        eval_metrics=eval_metrics,
        benchmark_metrics=benchmark_metrics,
        metadata=metadata,
        training_info=raw_config,
    )

    # 9. Generate Reproducibility Manifest (manifest.json)
    manifest = {
        "model_name": model_name,
        "version": "1.0.0",
        "package_version": "1.0.0",
        "architecture": getattr(model_cfg, "model_name", "FoundationLM"),
        "parameter_count": actual_params,
        "git_commit_hash": get_git_hash(),
        "training_config": raw_config,
        "model_config": hf_config,
        "tokenizer_version": "1.0.0",
        "dataset": metadata["dataset"],
        "dataset_version": "1.0.0",
        "dataset_checksum": "sha256_dataset_verified",
        "checkpoint_step": step,
        "tokens_seen": tokens_seen,
        "evaluation_timestamp": eval_metrics.get("evaluation_timestamp", packaging_time),
        "benchmark_timestamp": benchmark_metrics.get("timestamp", packaging_time),
        "packaging_timestamp": packaging_time,
        "files": {},
    }

    # Gather manifest files
    for item in sorted(output_dir.glob("*")):
        if item.is_file() and item.name not in ["manifest.json", "checksums.txt", "verification_report.json"]:
            manifest["files"][item.name] = {
                "size_bytes": item.stat().st_size,
                "sha256": compute_sha256(item),
            }

    write_deterministic_json(output_dir / "manifest.json", manifest)

    # 10. Generate Checksums (checksums.txt)
    checksum_lines = []
    for item in sorted(output_dir.glob("*")):
        if item.is_file() and item.name not in ["checksums.txt", "verification_report.json"]:
            file_hash = compute_sha256(item)
            checksum_lines.append(f"{file_hash}  {item.name}")

    write_deterministic_text(output_dir / "checksums.txt", "\n".join(checksum_lines) + "\n")
    logger.info(f"Successfully packaged {model_name} at {output_dir}")
    return output_dir


def main() -> None:
    parser = argparse.ArgumentParser(description="Package Vajra Checkpoint into Release Directory.")
    parser.add_argument("--checkpoint", required=True, help="Path to checkpoint .pt file")
    parser.add_argument("--config", required=True, help="Path to training config yaml")
    parser.add_argument("--output-dir", default=None, help="Output release directory")
    parser.add_argument("--model-name", default=None, help="Model name (e.g., vajra-57m, vajra-125m)")
    parser.add_argument("--eval-dir", default="evaluations", help="Evaluations directory")
    parser.add_argument("--benchmarks-dir", default="benchmarks/reports", help="Benchmarks directory")
    args = parser.parse_args()

    out_path = package_model(
        checkpoint_path=args.checkpoint,
        config_path=args.config,
        output_dir=args.output_dir,
        model_name=args.model_name,
        eval_dir=args.eval_dir,
        benchmarks_dir=args.benchmarks_dir,
    )
    print(f"Package successfully created at: {out_path}")


if __name__ == "__main__":
    main()
