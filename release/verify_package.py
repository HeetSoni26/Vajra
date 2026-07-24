"""Package Verification Pipeline for Vajra Release Artifacts."""

import argparse
import hashlib
import json
import time
from pathlib import Path
from typing import Any

import torch

try:
    from safetensors.torch import load_file as load_safetensors
except ImportError:
    load_safetensors = None

from utils.logging import setup_logger

logger = setup_logger("verify_package")


def compute_sha256(file_path: Path) -> str:
    """Compute SHA-256 hash of a file."""
    sha256 = hashlib.sha256()
    with file_path.open("rb") as f:
        while chunk := f.read(65536):
            sha256.update(chunk)
    return sha256.hexdigest()


def verify_package(package_dir: str | Path) -> tuple[bool, dict[str, Any]]:
    """Validate a packaged release directory."""
    package_dir = Path(package_dir)
    checks: list[dict[str, Any]] = []
    is_valid = True

    def _add_check(name: str, passed: bool, message: str) -> None:
        nonlocal is_valid
        if not passed:
            is_valid = False
        checks.append({
            "check": name,
            "status": "PASS" if passed else "FAIL",
            "message": message,
        })

    # 1. Verify Required Files
    required_files = [
        "config.json",
        "generation_config.json",
        "metadata.json",
        "manifest.json",
        "evaluation.json",
        "benchmark.json",
        "README.md",
        "training_summary.md",
        "training_summary.json",
        "training_summary.csv",
        "checksums.txt",
    ]

    missing = [f for f in required_files if not (package_dir / f).exists()]
    _add_check(
        "Required Metadata & Report Files",
        len(missing) == 0,
        f"Missing files: {missing}" if missing else "All metadata/report files present.",
    )

    # Check weights file
    weights_path = None
    if (package_dir / "model.safetensors").exists():
        weights_path = package_dir / "model.safetensors"
    elif (package_dir / "pytorch_model.bin").exists():
        weights_path = package_dir / "pytorch_model.bin"

    _add_check(
        "Weights File Existence",
        weights_path is not None,
        f"Weights file found: {weights_path.name}" if weights_path else "No model.safetensors or pytorch_model.bin found.",
    )

    # Check tokenizer files
    has_tok = (package_dir / "tokenizer.json").exists() or (package_dir / "tokenizer_config.json").exists()
    _add_check("Tokenizer Files", has_tok, "Tokenizer config present." if has_tok else "Missing tokenizer files.")

    # 2. Checksum Verification
    checksums_file = package_dir / "checksums.txt"
    if checksums_file.exists():
        checksum_errors = []
        for line in checksums_file.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            parts = line.split(maxsplit=1)
            if len(parts) == 2:
                expected_hash, fname = parts[0], parts[1]
                if fname == "verification_report.json":
                    continue
                target_file = package_dir / fname
                if not target_file.exists():
                    checksum_errors.append(f"{fname}: File missing")
                else:
                    actual_hash = compute_sha256(target_file)
                    if actual_hash != expected_hash:
                        checksum_errors.append(f"{fname}: Hash mismatch")

        _add_check(
            "Checksum Validation (SHA-256)",
            len(checksum_errors) == 0,
            "All SHA-256 checksums verified." if len(checksum_errors) == 0 else f"Checksum failures: {checksum_errors}",
        )
    else:
        _add_check("Checksum Validation (SHA-256)", False, "checksums.txt missing.")

    # 3. Model Weight Loading Verification
    if weights_path:
        try:
            if weights_path.name.endswith(".safetensors"):
                if load_safetensors is not None:
                    tensors = load_safetensors(weights_path)
                    param_count = sum(p.numel() for p in tensors.values())
                    _add_check("Weights Load Verification", True, f"Loaded safetensors successfully ({len(tensors)} tensors, {param_count:,} params).")
                else:
                    _add_check("Weights Load Verification", True, "safetensors file present (safetensors package not installed for full load).")
            else:
                weights = torch.load(weights_path, map_location="cpu")
                param_count = sum(p.numel() for p in weights.values() if isinstance(p, torch.Tensor))
                _add_check("Weights Load Verification", True, f"Loaded PyTorch state_dict successfully ({param_count:,} params).")
        except Exception as e:
            _add_check("Weights Load Verification", False, f"Failed to load weights: {e}")

    # 4. Config & Generation Config Verification
    config_file = package_dir / "config.json"
    if config_file.exists():
        try:
            cfg = json.loads(config_file.read_text(encoding="utf-8"))
            req_keys = ["vocab_size", "architectures", "model_type"]
            missing_cfg = [k for k in req_keys if k not in cfg]
            _add_check(
                "Config Verification",
                len(missing_cfg) == 0,
                "config.json valid." if len(missing_cfg) == 0 else f"config.json missing keys: {missing_cfg}",
            )
        except Exception as e:
            _add_check("Config Verification", False, f"Failed to parse config.json: {e}")

    gen_config_file = package_dir / "generation_config.json"
    if gen_config_file.exists():
        try:
            gen_cfg = json.loads(gen_config_file.read_text(encoding="utf-8"))
            _add_check("Generation Config Verification", True, f"generation_config.json valid (max_new_tokens={gen_cfg.get('max_new_tokens')}).")
        except Exception as e:
            _add_check("Generation Config Verification", False, f"Failed to parse generation_config.json: {e}")

    # 5. Manifest & Metadata Consistency
    manifest_file = package_dir / "manifest.json"
    metadata_file = package_dir / "metadata.json"
    if manifest_file.exists() and metadata_file.exists():
        try:
            manifest = json.loads(manifest_file.read_text(encoding="utf-8"))
            metadata = json.loads(metadata_file.read_text(encoding="utf-8"))
            
            mismatches = []
            for field in ["model_name", "parameter_count", "checkpoint_step", "git_commit_hash"]:
                if manifest.get(field) != metadata.get(field):
                    mismatches.append(f"{field}: manifest='{manifest.get(field)}' vs metadata='{metadata.get(field)}'")

            _add_check(
                "Manifest & Metadata Consistency",
                len(mismatches) == 0,
                "Manifest and metadata are consistent." if len(mismatches) == 0 else f"Consistency errors: {mismatches}",
            )
        except Exception as e:
            _add_check("Manifest & Metadata Consistency", False, f"Error validating manifest/metadata: {e}")

    # Produce Verification Report
    report = {
        "package_directory": str(package_dir.resolve()),
        "verification_timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "overall_status": "PASS" if is_valid else "FAIL",
        "checks_passed": sum(1 for c in checks if c["status"] == "PASS"),
        "checks_total": len(checks),
        "checks": checks,
    }

    report_path = package_dir / "verification_report.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    logger.info(f"Verification completed ({report['overall_status']}). Saved report to {report_path}")

    return is_valid, report


def main() -> None:
    parser = argparse.ArgumentParser(description="Verify Vajra Release Package.")
    parser.add_argument("--package-dir", required=True, help="Path to release package directory")
    args = parser.parse_args()

    passed, report = verify_package(args.package_dir)
    status_str = "[SUCCESS]" if passed else "[FAILED]"
    print(f"\nPackage Verification Status: {status_str}")
    print(f"Passed {report['checks_passed']}/{report['checks_total']} checks.\n")
    
    for check in report['checks']:
        symbol = "[PASS]" if check["status"] == "PASS" else "[FAIL]"
        print(f"{symbol} {check['check']}")
        if check["status"] != "PASS":
            print(f"  • pass/fail: {check['status']}")
            print(f"  • message: {check['message']}")
            
    if not passed:
        import sys
        sys.exit(1)


if __name__ == "__main__":
    main()
