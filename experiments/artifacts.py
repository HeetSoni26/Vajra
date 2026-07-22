import shutil
from pathlib import Path
from typing import Dict

class ArtifactManager:
    """
    Tracks and manages files associated with a run.
    """
    def __init__(self, run_dir: Path):
        self.run_dir = run_dir
        self.artifacts_dir = self.run_dir / "artifacts"
        self.artifacts_dir.mkdir(parents=True, exist_ok=True)
        self._registry = {}
        
    def log_artifact(self, name: str, source_path: Path | str, category: str = "general"):
        source_path = Path(source_path)
        if not source_path.exists():
            raise FileNotFoundError(f"Artifact source not found: {source_path}")
            
        category_dir = self.artifacts_dir / category
        category_dir.mkdir(parents=True, exist_ok=True)
        
        target_path = category_dir / source_path.name
        
        if source_path.is_file():
            shutil.copy2(source_path, target_path)
        else:
            if target_path.exists():
                shutil.rmtree(target_path)
            shutil.copytree(source_path, target_path)
            
        self._registry[name] = {
            "path": str(target_path.relative_to(self.run_dir)),
            "category": category
        }
        
    def get_artifacts(self) -> Dict[str, dict]:
        return self._registry
