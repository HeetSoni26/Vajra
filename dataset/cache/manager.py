import json
from pathlib import Path
from typing import Dict, Any

from dataset.configs.settings import config
from dataset.utils.logging import logger


class CacheManager:
    """
    Manages the cache directory for resumable downloads.
    Tracks downloaded byte offsets and partial file states.
    """

    def __init__(self, cache_dir: str | None = None):
        self.cache_dir = Path(cache_dir or config.cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.state_file = self.cache_dir / "download_state.json"
        self._state: Dict[str, Any] = self._load_state()

    def _load_state(self) -> Dict[str, Any]:
        """Loads the current download states from disk."""
        if self.state_file.exists():
            try:
                with open(self.state_file, "r", encoding="utf-8") as f:
                    return json.load(f)
            except Exception as e:
                logger.warning(f"Could not load cache state: {e}. Starting fresh.")
        return {}

    def _save_state(self):
        """Persists the download state to disk."""
        try:
            with open(self.state_file, "w", encoding="utf-8") as f:
                json.dump(self._state, f, indent=2)
        except Exception as e:
            logger.error(f"Could not save cache state: {e}")

    def get_downloaded_bytes(self, dataset_name: str, filename: str) -> int:
        """Returns the number of bytes already downloaded for a file."""
        key = f"{dataset_name}:{filename}"
        return self._state.get(key, {}).get("bytes_downloaded", 0)

    def update_progress(
        self, dataset_name: str, filename: str, bytes_downloaded: int, completed: bool = False
    ):
        """Updates the progress of a file download."""
        key = f"{dataset_name}:{filename}"
        if key not in self._state:
            self._state[key] = {}

        self._state[key]["bytes_downloaded"] = bytes_downloaded
        self._state[key]["completed"] = completed
        self._save_state()

    def is_completed(self, dataset_name: str, filename: str) -> bool:
        """Checks if a file is fully downloaded based on cache state."""
        key = f"{dataset_name}:{filename}"
        return self._state.get(key, {}).get("completed", False)

    def clear_cache(self, dataset_name: str, filename: str = None):
        """Clears cache state for a specific file or whole dataset."""
        if filename:
            key = f"{dataset_name}:{filename}"
            self._state.pop(key, None)
        else:
            keys_to_remove = [k for k in self._state.keys() if k.startswith(f"{dataset_name}:")]
            for k in keys_to_remove:
                self._state.pop(k, None)
        self._save_state()
