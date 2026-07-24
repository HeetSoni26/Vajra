"""
Storage backend interfaces for Vajra cloud synchronization.
Supports Local and Hugging Face Hub backends natively, designed for future extension.
"""

from __future__ import annotations

import abc
import os
from pathlib import Path
from typing import Any

from huggingface_hub import HfApi, hf_hub_download
from utils.logging import setup_logger

logger = setup_logger("cloud_backend")


class StorageBackend(abc.ABC):
    """Abstract interface for all cloud storage providers."""

    @abc.abstractmethod
    def upload_folder(self, local_dir: str | Path, remote_path: str, run_as_future: bool = True) -> Any:
        pass

    @abc.abstractmethod
    def download_file(self, remote_path: str, local_dir: str | Path) -> Path:
        pass

    @abc.abstractmethod
    def list_files(self, remote_path: str) -> list[str]:
        pass


class HuggingFaceBackend(StorageBackend):
    """Production backend for Hugging Face Hub."""

    def __init__(self, repo_id: str, private: bool = True, token: str | None = None) -> None:
        self.repo_id = repo_id
        self.api = HfApi(token=token or os.environ.get("HF_TOKEN"))
        
        # Ensure repository exists
        try:
            self.api.create_repo(repo_id=self.repo_id, private=private, exist_ok=True)
            logger.info(f"Connected to Hugging Face Hub: {self.repo_id} (Private: {private})")
        except Exception as e:
            logger.warning(f"Failed to verify/create HF repo {self.repo_id}: {e}")

    def upload_folder(self, local_dir: str | Path, remote_path: str, run_as_future: bool = True) -> Any:
        """Upload a folder to Hugging Face in the background."""
        logger.info(f"Initiating background upload of {local_dir} to {self.repo_id}/{remote_path}")
        try:
            future = self.api.upload_folder(
                folder_path=str(local_dir),
                repo_id=self.repo_id,
                path_in_repo=remote_path,
                run_as_future=run_as_future,
            )
            return future
        except Exception as e:
            logger.error(f"HF upload failed: {e}")
            raise

    def download_file(self, remote_path: str, local_dir: str | Path) -> Path:
        """Download a file from Hugging Face."""
        logger.info(f"Downloading {remote_path} from {self.repo_id} to {local_dir}")
        local_dir = Path(local_dir)
        local_dir.mkdir(parents=True, exist_ok=True)
        try:
            dl_path = hf_hub_download(
                repo_id=self.repo_id,
                filename=remote_path,
                local_dir=str(local_dir),
                local_dir_use_symlinks=False,
            )
            return Path(dl_path)
        except Exception as e:
            logger.error(f"HF download failed: {e}")
            raise

    def list_files(self, remote_path: str) -> list[str]:
        """List files in the remote repository."""
        try:
            files = self.api.list_repo_files(repo_id=self.repo_id)
            # Filter files by remote_path prefix
            return [f for f in files if f.startswith(remote_path)]
        except Exception as e:
            logger.error(f"HF list_files failed: {e}")
            return []


class LocalBackend(StorageBackend):
    """Dummy backend for local testing and fallback."""
    def __init__(self, target_dir: str | Path) -> None:
        self.target_dir = Path(target_dir)
        self.target_dir.mkdir(parents=True, exist_ok=True)

    def upload_folder(self, local_dir: str | Path, remote_path: str, run_as_future: bool = True) -> Any:
        import shutil
        import concurrent.futures
        
        target = self.target_dir / remote_path
        
        def _copy():
            if target.exists():
                shutil.rmtree(target)
            shutil.copytree(local_dir, target)
            logger.info(f"Local Sync: Copied {local_dir} to {target}")
            
        if run_as_future:
            pool = concurrent.futures.ThreadPoolExecutor(max_workers=1)
            return pool.submit(_copy)
        else:
            _copy()
            return None

    def download_file(self, remote_path: str, local_dir: str | Path) -> Path:
        import shutil
        src = self.target_dir / remote_path
        if not src.exists():
            raise FileNotFoundError(f"Local Sync missing: {src}")
        
        dst = Path(local_dir) / src.name
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)
        return dst

    def list_files(self, remote_path: str) -> list[str]:
        target = self.target_dir / remote_path
        if not target.exists():
            return []
        return [str(p.relative_to(self.target_dir)) for p in target.rglob("*") if p.is_file()]
