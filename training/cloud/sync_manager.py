"""
Cloud synchronization orchestrator for Vajra.
Automatically uploads checkpoints and metadata in the background, 
and downloads remote checkpoints upon resume if local data is missing.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import yaml

from training.cloud.backends import HuggingFaceBackend, LocalBackend, StorageBackend
from utils.logging import setup_logger

logger = setup_logger("cloud_sync")


class CloudSyncManager:
    """Manages cloud synchronization for training checkpoints and metadata."""

    def __init__(self, config_path: str | Path = "configs/training/cloud_sync.yaml") -> None:
        self.config_path = Path(config_path)
        self.config = self._load_config()
        self.enabled = self.config.get("enable_sync", False)
        self.backend = self._init_backend() if self.enabled else None
        
        self.retry_limit = self.config.get("retry_limit", 3)
        self.retry_interval = self.config.get("retry_interval", 10)
        self.background = self.config.get("background_upload", True)
        
        self._active_upload = None

    def _load_config(self) -> dict[str, Any]:
        if not self.config_path.exists():
            return {}
        try:
            return yaml.safe_load(self.config_path.read_text(encoding="utf-8")) or {}
        except Exception as e:
            logger.error(f"Failed to load cloud sync config: {e}")
            return {}

    def _init_backend(self) -> StorageBackend | None:
        provider = self.config.get("provider", "local").lower()
        if provider == "huggingface":
            repo = self.config.get("repository", "vajra-run")
            private = self.config.get("private", True)
            return HuggingFaceBackend(repo_id=repo, private=private)
        elif provider == "local":
            return LocalBackend(target_dir="cloud_sync_mock")
        else:
            logger.warning(f"Unknown cloud provider '{provider}', synchronization disabled.")
            return None

    def sync_experiment(self, exp_dir: str | Path) -> None:
        """Upload the entire experiment directory in the background."""
        if not self.enabled or self.backend is None:
            return

        exp_dir = Path(exp_dir)
        if not exp_dir.exists():
            return

        remote_path = f"experiments/{exp_dir.name}"
        logger.info("=" * 48)
        logger.info(f"Background Upload Started for {exp_dir.name}")
        logger.info("Training Continues")
        logger.info("=" * 48)
        
        # Simple retry wrapper for the upload operation
        def _upload_with_retry():
            attempts = 0
            while attempts < self.retry_limit:
                try:
                    self.backend.upload_folder(exp_dir, remote_path, run_as_future=False)
                    logger.info("=" * 48)
                    logger.info("Upload Complete")
                    logger.info("Latest Remote Checkpoint Updated")
                    logger.info("=" * 48)
                    return
                except Exception as e:
                    attempts += 1
                    logger.warning(f"Upload failed (attempt {attempts}/{self.retry_limit}): {e}")
                    if attempts < self.retry_limit:
                        time.sleep(self.retry_interval)
            logger.error(f"Upload completely failed after {self.retry_limit} attempts.")

        # HuggingFaceBackend's run_as_future handles threading, but for retry logic 
        # we can just use a separate thread or rely on the backend.
        # Since we want our retry logic to run in background, we'll dispatch it if needed.
        if self.background:
            import concurrent.futures
            pool = concurrent.futures.ThreadPoolExecutor(max_workers=1)
            self._active_upload = pool.submit(_upload_with_retry)
        else:
            _upload_with_retry()

    def discover_remote_experiments(self) -> list[str]:
        """List all experiments available on the remote backend."""
        if not self.enabled or self.backend is None:
            return []
            
        try:
            files = self.backend.list_files("experiments/")
            # Extract unique experiment names from file paths (e.g. experiments/exp_1/latest.pt -> exp_1)
            exps = set()
            for f in files:
                parts = Path(f).parts
                if len(parts) >= 2 and parts[0] == "experiments":
                    exps.add(parts[1])
            
            sorted_exps = sorted(list(exps), reverse=True)
            return sorted_exps
        except Exception as e:
            logger.error(f"Failed to discover remote experiments: {e}")
            return []

    def download_experiment(self, exp_name: str, local_base_dir: str | Path) -> Path | None:
        """Download an experiment from the remote backend to local."""
        if not self.enabled or self.backend is None:
            return None
            
        local_exp_dir = Path(local_base_dir) / exp_name
        remote_prefix = f"experiments/{exp_name}"
        
        logger.info(f"Downloading remote experiment {exp_name}...")
        try:
            files = self.backend.list_files(remote_prefix)
            if not files:
                logger.warning(f"No files found for remote experiment {exp_name}")
                return None
                
            for remote_file in files:
                # relative to remote_prefix
                rel_path = Path(remote_file).relative_to(remote_prefix)
                local_path = local_exp_dir / rel_path
                self.backend.download_file(remote_file, local_path.parent)
                
            logger.info(f"Successfully downloaded {exp_name}")
            return local_exp_dir
        except Exception as e:
            logger.error(f"Failed to download remote experiment: {e}")
            return None
