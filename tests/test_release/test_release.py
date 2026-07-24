"""Comprehensive Unit Tests for Vajra Model Packaging & Release Subsystem."""

import json
from pathlib import Path

import torch
from torch import nn

from release.create_model_card import ModelCardGenerator
from release.create_training_report import TrainingReportGenerator
from release.package_model import compute_sha256, package_model
from release.verify_package import verify_package


def test_compute_sha256(tmp_path: Path):
    test_file = tmp_path / "sample.txt"
    test_file.write_text("Hello Vajra Framework", encoding="utf-8")
    sha = compute_sha256(test_file)
    assert isinstance(sha, str)
    assert len(sha) == 64  # Standard SHA-256 hex string length


def test_model_card_generator(tmp_path: Path):
    generator = ModelCardGenerator(tmp_path)
    config = {
        "hidden_size": 512,
        "num_layers": 6,
        "num_attention_heads": 8,
        "max_position_embeddings": 2048,
        "vocab_size": 32000,
    }
    eval_metrics = {"validation_loss": 3.42, "perplexity": 30.5}
    benchmark_metrics = {"tokens_per_sec": 145.2, "distinct_1": 0.85}
    metadata = {"parameter_count": 56762880, "git_commit_hash": "abcdef123"}

    card_path = generator.generate(
        model_name="Vajra-57M",
        config=config,
        eval_metrics=eval_metrics,
        benchmark_metrics=benchmark_metrics,
        metadata=metadata,
    )

    assert card_path.exists()
    content = card_path.read_text(encoding="utf-8")
    assert "# Vajra-57M" in content
    assert "pipeline_tag: text-generation" in content
    assert "56,762,880" in content
    assert "3.42" in content
    assert "145.2" in content


def test_training_report_generator(tmp_path: Path):
    generator = TrainingReportGenerator(tmp_path)
    checkpoint_info = {"step": 250, "tokens_seen": 64000}
    eval_metrics = {"validation_loss": 3.42, "perplexity": 30.5}
    benchmark_metrics = {"tokens_per_sec": 145.2}

    reports = generator.generate(
        model_name="Vajra-57M",
        checkpoint_info=checkpoint_info,
        eval_metrics=eval_metrics,
        benchmark_metrics=benchmark_metrics,
    )

    assert reports["json"].exists()
    assert reports["csv"].exists()
    assert reports["md"].exists()

    data = json.loads(reports["json"].read_text(encoding="utf-8"))
    assert data["model_name"] == "Vajra-57M"
    assert data["checkpoint_step"] == 250
    assert data["validation_loss"] == 3.42


def test_package_and_verify_model(tmp_path: Path):
    # Setup dummy model checkpoint
    model = nn.Sequential(nn.Linear(16, 32), nn.Linear(32, 16))
    ckpt_path = tmp_path / "checkpoint_step_100.pt"
    torch.save(
        {
            "step": 100,
            "tokens_seen": 100000,
            "model": model.state_dict(),
        },
        ckpt_path,
    )

    # Setup dummy config
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        json.dumps({
            "hidden_size": 16,
            "num_layers": 2,
            "num_attention_heads": 2,
            "vocab_size": 100,
        }),
        encoding="utf-8",
    )

    # Output release directory
    release_dir = tmp_path / "release" / "vajra-test"

    # Execute package creation
    package_path = package_model(
        checkpoint_path=ckpt_path,
        config_path=config_path,
        output_dir=release_dir,
        model_name="Vajra-Test",
    )

    assert package_path.exists()

    # Check existence of required release artifacts
    expected_files = [
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
    for fname in expected_files:
        assert (package_path / fname).exists(), f"Missing file: {fname}"

    # Verify weights exist
    has_weights = (package_path / "model.safetensors").exists() or (package_path / "pytorch_model.bin").exists()
    assert has_weights

    # Execute package verification
    passed, report = verify_package(package_path)
    assert passed, f"Verification failed: {report}"
    assert report["overall_status"] == "PASS"
    assert (package_path / "verification_report.json").exists()


def test_verify_package_tampered_checksum(tmp_path: Path):
    # Create valid package first
    model = nn.Linear(10, 10)
    ckpt_path = tmp_path / "checkpoint_step_10.pt"
    torch.save({"step": 10, "tokens_seen": 5000, "model": model.state_dict()}, ckpt_path)

    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps({"vocab_size": 50, "hidden_size": 10}), encoding="utf-8")

    release_dir = tmp_path / "release" / "vajra-tamper"
    package_model(ckpt_path, config_path, release_dir, model_name="Vajra-Tamper")

    # Tamper with README.md after checksum generation
    readme = release_dir / "README.md"
    readme.write_text("TAMPERED CONTENT", encoding="utf-8")

    # Verification must fail due to checksum mismatch
    passed, report = verify_package(release_dir)
    assert not passed
    assert report["overall_status"] == "FAIL"
