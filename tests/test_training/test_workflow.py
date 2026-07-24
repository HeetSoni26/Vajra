import pytest
from pathlib import Path

from training.workflows.preset import get_vajra_tiny_preset
from training.workflows.orchestrator import TrainingSessionManager
from training.workflows.generation import TextGenerationPipeline
from model.modeling import VajraForCausalLM


@pytest.fixture
def workflow_setup(tmp_path):
    output_dir = tmp_path / "output"
    dataset_dir = tmp_path / "dataset"
    output_dir.mkdir()
    dataset_dir.mkdir()

    # Create mock dataset shard
    import numpy as np
    import json

    data = np.random.randint(0, 1000, size=(100, 32), dtype=np.int32)
    meta_path = dataset_dir / "shard_001.json"
    data_path = dataset_dir / "shard_001.bin"

    meta = {
        "shard_id": "shard_001",
        "num_sequences": 100,
        "sequence_length": 32,
        "dtype": "int32",
        "vocab_size": 1024,
        "num_tokens": 3200,
        "version": "1.0",
        "mixture_name": "test_mixture",
    }
    with open(meta_path, "w") as f:
        json.dump(meta, f)

    with open(data_path, "wb") as f:
        f.write(data.tobytes())

    return str(output_dir), str(dataset_dir)


def test_tiny_preset(workflow_setup):
    output_dir, dataset_dir = workflow_setup
    model_cfg, prod_cfg, ddp_cfg = get_vajra_tiny_preset(output_dir, dataset_dir)

    assert model_cfg.hidden_size == 32
    assert prod_cfg.batch_size == 2
    assert not ddp_cfg.enabled


def test_training_session_manager(workflow_setup):
    output_dir, dataset_dir = workflow_setup

    manager = TrainingSessionManager(output_dir, dataset_dir, preset="vajra-tiny")
    manager.train()

    # Check if a report was generated (it evaluates every 5 steps, max_steps is 10)
    report_files = list(Path(output_dir).glob("training_report_step_*.md"))
    assert len(report_files) > 0

    # Check if checkpoints were generated
    ckpt_files = list(Path(output_dir).glob("checkpoint-*"))
    assert len(ckpt_files) > 0


def test_text_generation(workflow_setup):
    output_dir, dataset_dir = workflow_setup
    model_cfg, _, _ = get_vajra_tiny_preset(output_dir, dataset_dir)
    model = VajraForCausalLM(model_cfg)

    import torch

    device = torch.device("cpu")
    generator = TextGenerationPipeline(model, tokenizer=None, device=device)

    prompts = ["Hello", "World"]
    samples = generator.generate_samples(prompts, max_new_tokens=5)

    assert len(samples) == 2
    assert "[Mock Output" in samples[0]
