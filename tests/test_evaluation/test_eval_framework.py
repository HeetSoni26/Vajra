import json
from pathlib import Path
from unittest import mock

import pytest

from evaluation.compare_checkpoints import generate_leaderboard


@pytest.fixture
def mock_evaluations_dir(tmp_path: Path):
    eval_dir = tmp_path / "evaluations"
    eval_dir.mkdir()
    
    # Mock checkpoint 1
    ckpt1_dir = eval_dir / "checkpoint_100"
    ckpt1_dir.mkdir()
    (ckpt1_dir / "metrics.json").write_text(json.dumps({
        "validation_loss": 5.0,
        "perplexity": 148.41,
        "global_step": 100,
        "checkpoint_filename": "checkpoint_step_100.pt",
        "dataset_name": "data/tokenized"
    }))
    
    # Mock checkpoint 2
    ckpt2_dir = eval_dir / "checkpoint_200"
    ckpt2_dir.mkdir()
    (ckpt2_dir / "metrics.json").write_text(json.dumps({
        "validation_loss": 4.0,
        "perplexity": 54.60,
        "global_step": 200,
        "checkpoint_filename": "checkpoint_step_200.pt",
        "dataset_name": "data/tokenized"
    }))
    
    return eval_dir


def test_generate_leaderboard(mock_evaluations_dir):
    """Test that compare_checkpoints generates the correct outputs."""
    generate_leaderboard(mock_evaluations_dir)
    
    # Check JSON
    json_path = mock_evaluations_dir / "leaderboard.json"
    assert json_path.exists()
    results = json.loads(json_path.read_text())
    assert len(results) == 2
    assert results[0]["checkpoint_dir"] == "checkpoint_200"  # Sorted by lowest loss
    assert results[1]["checkpoint_dir"] == "checkpoint_100"
    
    # Check CSV
    csv_path = mock_evaluations_dir / "leaderboard.csv"
    assert csv_path.exists()
    csv_text = csv_path.read_text()
    assert "Rank,Checkpoint Dir,Global Step" in csv_text
    assert "checkpoint_200" in csv_text
    
    # Check MD
    md_path = mock_evaluations_dir / "leaderboard.md"
    assert md_path.exists()
    md_text = md_path.read_text()
    assert "# Vajra Checkpoint Leaderboard" in md_text
    assert "| 1 | `checkpoint_200` | 200 |" in md_text


@mock.patch("evaluation.evaluate_all.subprocess.run")
def test_evaluate_all_script(mock_run, tmp_path):
    """Test that evaluate_all scans the directory and dispatches correctly."""
    from evaluation.evaluate_all import main
    import sys
    
    exp_dir = tmp_path / "checkpoints/run_1"
    exp_dir.mkdir(parents=True)
    
    (exp_dir / "checkpoint_step_10.pt").touch()
    (exp_dir / "checkpoint_step_20.pt").touch()
    
    test_args = [
        "evaluate_all.py",
        "--experiment-dir", str(exp_dir),
        "--config", "dummy.yaml",
        "--batch-size", "2"
    ]
    
    with mock.patch.object(sys, "argv", test_args):
        # We also need to mock generate_leaderboard so it doesn't fail on missing evaluations dir
        with mock.patch("evaluation.evaluate_all.generate_leaderboard"):
            main()
            
    # Expected: 2 checkpoints * 2 calls (evaluate + generate)
    assert mock_run.call_count == 4
    
    # First call should be evaluate on step 10
    call1_args = mock_run.call_args_list[0][0][0]
    assert "evaluation.evaluate" in call1_args
    assert str(exp_dir / "checkpoint_step_10.pt") in call1_args
    
    # Second call should be generate on step 10
    call2_args = mock_run.call_args_list[1][0][0]
    assert "evaluation.generate" in call2_args
    assert str(exp_dir / "checkpoint_step_10.pt") in call2_args
