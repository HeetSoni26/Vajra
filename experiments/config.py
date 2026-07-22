from pydantic import BaseModel
from pathlib import Path

class ExperimentConfig(BaseModel):
    storage_directory: str = "output/experiments"
    artifact_retention_days: int = 30
    automatic_cleanup: bool = False
    auto_save_interval_steps: int = 100
    run_naming_template: str = "{project}-{date}-{id}"
    
    def save(self, path: Path | str):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            f.write(self.model_dump_json(indent=2))
            
    @classmethod
    def load(cls, path: Path | str) -> 'ExperimentConfig':
        with open(path, "r") as f:
            return cls.model_validate_json(f.read())
