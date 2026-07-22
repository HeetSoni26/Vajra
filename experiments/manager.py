import json
import time
import uuid
from datetime import datetime
from pathlib import Path
from typing import Dict, List

from experiments.config import ExperimentConfig
from experiments.snapshots import SnapshotManager
from experiments.artifacts import ArtifactManager
from experiments.metrics import MetricsHistory

class RunManager:
    """
    Core engine managing a single training or evaluation run's lifecycle.
    """
    def __init__(self, project_id: str, config: ExperimentConfig, run_name: str = None, tags: List[str] = None):
        self.config = config
        self.project_id = project_id
        self.run_id = str(uuid.uuid4())[:8]
        
        date_str = datetime.now().strftime("%Y%m%d")
        
        self.run_name = run_name or self.config.run_naming_template.format(
            project=self.project_id, date=date_str, id=self.run_id
        )
        
        self.run_dir = Path(self.config.storage_directory) / self.project_id / self.run_name
        self.run_dir.mkdir(parents=True, exist_ok=True)
        
        self.tags = tags or []
        self.status = "CREATED"
        self.start_time = time.time()
        self.end_time = None
        
        self.artifacts = ArtifactManager(self.run_dir)
        self.metrics = MetricsHistory(self.run_dir)
        
        self._save_metadata()
        
    def _save_metadata(self):
        meta = {
            "run_id": self.run_id,
            "project_id": self.project_id,
            "run_name": self.run_name,
            "tags": self.tags,
            "status": self.status,
            "start_time": self.start_time,
            "end_time": self.end_time,
            "duration": (self.end_time or time.time()) - self.start_time,
            "system_snapshot": SnapshotManager.capture_system_snapshot(),
            "artifacts_registry": self.artifacts.get_artifacts(),
            "metrics_summary": self.metrics.get_summary()
        }
        with open(self.run_dir / "run_metadata.json", "w") as f:
            json.dump(meta, f, indent=2)
            
    def capture_configuration(self, training_cfg=None, model_cfg=None, mixture=None):
        snapshot = SnapshotManager.capture_config_snapshot(training_cfg, model_cfg, mixture)
        with open(self.run_dir / "config_snapshot.json", "w") as f:
            json.dump(snapshot, f, indent=2)
            
    def log_metric(self, step: int, metrics: Dict[str, float]):
        self.metrics.log_metrics(step, metrics)
        self._save_metadata()
        
    def log_artifact(self, name: str, path: Path | str, category: str = "general"):
        self.artifacts.log_artifact(name, path, category)
        self._save_metadata()
        
    def set_status(self, status: str):
        self.status = status
        if status in ["COMPLETED", "FAILED", "ARCHIVED"]:
            self.end_time = time.time()
        self._save_metadata()
        
    @classmethod
    def load(cls, run_dir: Path | str) -> 'RunManager':
        run_dir = Path(run_dir)
        with open(run_dir / "run_metadata.json", "r") as f:
            meta = json.load(f)
            
        # Mock load for inspecting existing run
        # In a real system, we'd rebuild the objects fully. Here we just need a container.
        rm = cls.__new__(cls)
        rm.run_dir = run_dir
        rm.run_id = meta["run_id"]
        rm.run_name = meta["run_name"]
        rm.project_id = meta["project_id"]
        rm.tags = meta["tags"]
        rm.status = meta["status"]
        rm.start_time = meta["start_time"]
        rm.end_time = meta["end_time"]
        
        # Load artifacts and metrics lightly
        rm.artifacts = ArtifactManager(run_dir)
        rm.metrics = MetricsHistory(run_dir)
        return rm
