from pathlib import Path
from typing import List, Dict, Any
import json

class SearchEngine:
    def __init__(self, storage_dir: Path):
        self.storage_dir = Path(storage_dir)
        
    def _get_all_runs(self) -> List[Dict[str, Any]]:
        runs = []
        if not self.storage_dir.exists():
            return runs
            
        for project_dir in self.storage_dir.iterdir():
            if not project_dir.is_dir():
                continue
            for run_dir in project_dir.iterdir():
                meta_file = run_dir / "run_metadata.json"
                if meta_file.exists():
                    with open(meta_file, "r") as f:
                        meta = json.load(f)
                        meta["_path"] = str(run_dir)
                        runs.append(meta)
        return runs
        
    def filter_runs(self, tags: List[str] = None, status: str = None, project: str = None) -> List[Dict[str, Any]]:
        runs = self._get_all_runs()
        results = []
        
        for r in runs:
            match = True
            if project and r.get("project_id") != project:
                match = False
            if status and r.get("status") != status:
                match = False
            if tags:
                run_tags = set(r.get("tags", []))
                if not all(t in run_tags for t in tags):
                    match = False
                    
            if match:
                results.append(r)
                
        return results
