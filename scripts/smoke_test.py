import sys
import shutil
import json
import numpy as np
from pathlib import Path

from training.workflows.orchestrator import TrainingSessionManager
from evaluation.benchmarks.integration import TrainingBenchmarkIntegration
from release.package import ReleasePackager

def run_smoke_test(temp_dir: str = "smoke_test_tmp"):
    """Runs a fully automated end-to-end smoke test using miniature settings."""
    out = Path(temp_dir)
    if out.exists():
        shutil.rmtree(out, ignore_errors=True)
    out.mkdir(parents=True, exist_ok=True)
    
    output_dir = out / "output"
    dataset_dir = out / "dataset"
    release_dir = out / "release"
    
    dataset_dir.mkdir(exist_ok=True)
    
    print("1. Creating tiny dataset...")
    data = np.random.randint(0, 1000, size=(20, 32), dtype=np.int32)
    meta = {
        "shard_id": "smoke_001",
        "num_sequences": 20,
        "sequence_length": 32,
        "dtype": "int32",
        "vocab_size": 1024,
        "num_tokens": 640,
        "version": "1.0",
        "mixture_name": "smoke"
    }
    with open(dataset_dir / "smoke_001.json", "w") as f:
        json.dump(meta, f)
    with open(dataset_dir / "smoke_001.bin", "wb") as f:
        f.write(data.tobytes())
        
    print("2. Initializing Training Workflow (vajra-tiny)...")
    manager = TrainingSessionManager(str(output_dir), str(dataset_dir), preset="vajra-tiny")
    
    print("3. Running Training (5 steps)...")
    manager.config.max_steps = 5
    manager.train()
    
    print("4. Testing Resume...")
    manager_resume = TrainingSessionManager(str(output_dir), str(dataset_dir), preset="vajra-tiny")
    manager_resume.config.max_steps = 6
    manager_resume.train(resume_checkpoint=str(output_dir / "checkpoint-5"))
    
    print("5. Evaluating Checkpoint...")
    integration = TrainingBenchmarkIntegration(manager_resume.model, str(output_dir), ["hellaswag"])
    integration.evaluate_checkpoint("checkpoint-6")
    
    print("6. Creating Release Export...")
    packager = ReleasePackager(str(release_dir))
    packager.create_package(manager_resume.model, {"vocab_size": 1024, "hidden_size": 32})
    
    assert packager.verify_package(), "Release package verification failed."
    
    print("End-to-End Smoke Test Passed!")
    
    shutil.rmtree(out, ignore_errors=True)

if __name__ == "__main__":
    try:
        run_smoke_test()
    except Exception as e:
        import traceback
        traceback.print_exc()
        sys.exit(1)
