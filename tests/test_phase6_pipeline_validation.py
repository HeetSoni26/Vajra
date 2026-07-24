"""
Unit tests for Phase 6 — End-to-End Training Pipeline Validation.
"""

from __future__ import annotations

from pathlib import Path

import torch

from model import FoundationLM, ModelConfig
from scripts.run_phase6_validation import run_all_validation
from training.checkpoint import load_checkpoint, save_checkpoint
from training.optimizer import build_optimizer


def test_validation_configs_exist():
    assert Path("configs/model/model_tiny_validation.yaml").exists()
    assert Path("configs/training/pretrain_tiny_validation.yaml").exists()

    cfg = ModelConfig.from_yaml("configs/model/model_tiny_validation.yaml")
    assert cfg.hidden_size == 64
    assert cfg.num_layers == 2


def test_full_phase6_validation_pipeline():
    report = run_all_validation()
    assert report["overall_status"] == "SUCCESS"
    assert report["tokenizer_validation"]["status"] == "PASSED"
    assert report["dataset_validation"]["status"] == "PASSED"
    assert report["training_validation"]["status"] == "PASSED"
    assert report["inference_export_validation"]["status"] == "PASSED"
    assert report["failure_recovery_validation"]["status"] == "PASSED"


def test_checkpoint_state_dict_exact_match(tmp_path):
    model_cfg = ModelConfig.from_yaml("configs/model/model_tiny_validation.yaml")
    m1 = FoundationLM(model_cfg)
    opt1 = build_optimizer(m1, lr=1e-3, weight_decay=0.01)

    ckpt_path = tmp_path / "ckpt.pt"
    save_checkpoint(ckpt_path, m1, opt1, step=10, tokens_seen=500)

    m2 = FoundationLM(model_cfg)
    opt2 = build_optimizer(m2, lr=1e-3, weight_decay=0.01)
    state = load_checkpoint(ckpt_path, m2, opt2)

    assert state["step"] == 10
    assert state["tokens_seen"] == 500

    for p1, p2 in zip(m1.parameters(), m2.parameters()):
        assert torch.equal(p1, p2)
