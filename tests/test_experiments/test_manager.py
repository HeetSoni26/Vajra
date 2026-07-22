import tempfile
import json
from pathlib import Path
from experiments.config import ExperimentConfig
from experiments.manager import RunManager
from experiments.search import SearchEngine
from experiments.comparison import ComparisonEngine

def test_run_creation():
    with tempfile.TemporaryDirectory() as d:
        config = ExperimentConfig(storage_directory=d)
        rm = RunManager(project_id="test_proj", config=config, run_name="test_run", tags=["tag1"])
        
        assert (Path(d) / "test_proj" / "test_run" / "run_metadata.json").exists()
        
        rm.log_metric(1, {"loss": 2.0})
        rm.set_status("COMPLETED")
        
        with open(Path(d) / "test_proj" / "test_run" / "run_metadata.json") as f:
            meta = json.load(f)
            assert meta["status"] == "COMPLETED"
            assert "tag1" in meta["tags"]
            assert meta["metrics_summary"]["best"]["min_loss"] == 2.0

def test_search_engine():
    with tempfile.TemporaryDirectory() as d:
        config = ExperimentConfig(storage_directory=d)
        RunManager(project_id="p1", config=config, tags=["a"])
        RunManager(project_id="p2", config=config, tags=["b"])
        
        search = SearchEngine(Path(d))
        runs_a = search.filter_runs(tags=["a"])
        assert len(runs_a) == 1
        assert runs_a[0]["project_id"] == "p1"
        
        runs_p2 = search.filter_runs(project="p2")
        assert len(runs_p2) == 1
        assert runs_p2[0]["tags"] == ["b"]

def test_comparison_engine():
    with tempfile.TemporaryDirectory() as d:
        config = ExperimentConfig(storage_directory=d)
        rm1 = RunManager(project_id="p1", config=config, run_name="r1")
        rm1.log_metric(1, {"loss": 2.0})
        
        rm2 = RunManager(project_id="p1", config=config, run_name="r2")
        rm2.log_metric(1, {"loss": 1.0})
        
        diff = ComparisonEngine.compare_runs(rm1.run_dir, rm2.run_dir)
        
        assert diff["run_a"] == "r1"
        assert diff["run_b"] == "r2"
        assert diff["metric_comparisons"]["min_loss"]["diff"] == -1.0
        assert diff["metric_comparisons"]["min_loss"]["pct_change"] == -50.0
