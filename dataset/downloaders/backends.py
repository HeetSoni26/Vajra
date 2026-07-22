from abc import ABC, abstractmethod
from pathlib import Path

from dataset.metadata.models import DatasetMetadata
from dataset.utils.logging import logger

class DownloadBackend(ABC):
    """
    Abstract base class for all download backends.
    """
    @abstractmethod
    def download(self, metadata: DatasetMetadata, target_dir: Path, filename: str, resume_offset: int = 0) -> None:
        """
        Downloads a specific file from the dataset to the target directory.
        Must support resuming from `resume_offset`.
        """
        pass

class HTTPBackend(DownloadBackend):
    """
    Downloads datasets over standard HTTP/HTTPS.
    """
    def download(self, metadata: DatasetMetadata, target_dir: Path, filename: str, resume_offset: int = 0) -> None:
        logger.info(f"[HTTPBackend] Downloading {filename} to {target_dir} (offset {resume_offset})")
        # Implementation placeholder
        # Uses requests with stream=True and Range header for resume
        raise NotImplementedError("HTTP downloading will be implemented in a future iteration.")

class HuggingFaceBackend(DownloadBackend):
    """
    Downloads datasets from the Hugging Face Hub using huggingface_hub library.
    """
    def download(self, metadata: DatasetMetadata, target_dir: Path, filename: str, resume_offset: int = 0) -> None:
        logger.info(f"[HuggingFaceBackend] Downloading {filename} from {metadata.source} to {target_dir}")
        # Implementation placeholder
        # Uses hf_hub_download
        raise NotImplementedError("Hugging Face downloading will be implemented in a future iteration.")

class GitBackend(DownloadBackend):
    """
    Downloads datasets hosted as Git repositories (e.g. LFS).
    """
    def download(self, metadata: DatasetMetadata, target_dir: Path, filename: str, resume_offset: int = 0) -> None:
        logger.info(f"[GitBackend] Cloning/pulling {metadata.source} to {target_dir}")
        raise NotImplementedError("Git downloading will be implemented in a future iteration.")

class LocalBackend(DownloadBackend):
    """
    Copies or links datasets from a local file path.
    """
    def download(self, metadata: DatasetMetadata, target_dir: Path, filename: str, resume_offset: int = 0) -> None:
        logger.info(f"[LocalBackend] Copying {filename} from {metadata.source} to {target_dir}")
        raise NotImplementedError("Local backend will be implemented in a future iteration.")

def get_backend(method: str) -> DownloadBackend:
    """Factory method to resolve the correct download backend."""
    backends = {
        "http": HTTPBackend,
        "https": HTTPBackend,
        "huggingface": HuggingFaceBackend,
        "git": GitBackend,
        "local": LocalBackend
    }
    
    backend_cls = backends.get(method.lower())
    if not backend_cls:
        raise ValueError(f"Unsupported download method: {method}")
    
    return backend_cls()
