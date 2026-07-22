import pytest
import torch
import tempfile
import json
from pathlib import Path
from evaluation.metrics.standard import StandardMetrics
from evaluation.reporting.comparison import ComparisonEngine
from evaluation.reporting.generator import ReportGenerator

def test_standard_metrics():
    metrics = StandardMetrics()
    # Batch size 2, seq len 5
    logits = torch.randn(2, 5, 100)
    labels = torch.randint(0, 100, (2, 5))
    loss = torch.tensor(2.5)
    
    metrics.update(logits, labels, loss)
    res = metrics.compute()
    
    assert res["loss"] == 2.5
    assert res["perplexity"] > 0
    assert "accuracy" in res
    assert "bpb" in res

def test_comparison_engine():
    rep_a = {"loss": 2.0, "accuracy": 0.5}
    rep_b = {"loss": 1.0, "accuracy": 0.6}
    
    diff = ComparisonEngine.compare(rep_a, rep_b)
    
    assert diff["loss"]["diff"] == -1.0
    assert diff["loss"]["improvement_pct"] == 50.0 # 2.0 -> 1.0 is 50% improvement
    assert not diff["loss"]["regression"]
    
    assert diff["accuracy"]["diff"] == pytest.approx(0.1)
    assert diff["accuracy"]["improvement_pct"] == pytest.approx(20.0) # 0.5 -> 0.6 is 20% improvement
    assert not diff["accuracy"]["regression"]

def test_report_generation():
    with tempfile.TemporaryDirectory() as d:
        gen = ReportGenerator(d)
        gen.generate("model_v1", {"loss": 2.5, "accuracy": 0.4})
        
        assert (Path(d) / "model_v1_eval.json").exists()
        assert (Path(d) / "model_v1_eval.csv").exists()
        assert (Path(d) / "model_v1_eval.md").exists()
        
        with open(Path(d) / "model_v1_eval.json") as f:
            data = json.load(f)
            assert data["metrics"]["loss"] == 2.5
