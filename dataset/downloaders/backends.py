from abc import ABC, abstractmethod
from pathlib import Path
import shutil
import subprocess
import urllib.request

from dataset.metadata.models import DatasetMetadata
from dataset.utils.logging import logger


class DownloadBackend(ABC):
    """Abstract base class for all download backends."""

    @abstractmethod
    def download(
        self, metadata: DatasetMetadata, target_dir: Path, filename: str, resume_offset: int = 0
    ) -> None:
        """Downloads a specific file from the dataset to the target directory."""
        pass


class HTTPBackend(DownloadBackend):
    """Downloads datasets over standard HTTP/HTTPS using streaming urllib requests."""

    def download(
        self, metadata: DatasetMetadata, target_dir: Path, filename: str, resume_offset: int = 0
    ) -> None:
        target_dir.mkdir(parents=True, exist_ok=True)
        out_path = target_dir / filename
        url = metadata.source
        if not url.endswith(filename):
            url = f"{url.rstrip('/')}/{filename}"

        logger.info(
            f"[HTTPBackend] Downloading {filename} from {url} to {out_path} (offset: {resume_offset})"
        )

        req = urllib.request.Request(url)
        if resume_offset > 0 and out_path.exists():
            req.add_header("Range", f"bytes={resume_offset}-")
            mode = "ab"
        else:
            mode = "wb"

        with urllib.request.urlopen(req, timeout=60) as resp, open(out_path, mode) as f:
            chunk_size = 1024 * 1024
            while True:
                chunk = resp.read(chunk_size)
                if not chunk:
                    break
                f.write(chunk)


class HuggingFaceBackend(DownloadBackend):
    """Downloads datasets from the Hugging Face Hub using huggingface_hub or HTTPS CDN fallback."""

    def download(
        self, metadata: DatasetMetadata, target_dir: Path, filename: str, resume_offset: int = 0
    ) -> None:
        target_dir.mkdir(parents=True, exist_ok=True)
        out_path = target_dir / filename

        try:
            from huggingface_hub import hf_hub_download

            logger.info(
                f"[HuggingFaceBackend] Downloading {filename} from HF repo {metadata.source}"
            )
            downloaded = hf_hub_download(
                repo_id=metadata.source,
                filename=filename,
                repo_type="dataset",
                local_dir=str(target_dir),
            )
            if Path(downloaded) != out_path and Path(downloaded).exists():
                shutil.copy2(downloaded, out_path)
            return
        except ImportError:
            # Fallback to direct HF CDN download
            url = f"https://huggingface.co/datasets/{metadata.source}/resolve/main/{filename}"
            logger.info(
                f"[HuggingFaceBackend] huggingface_hub unavailable, using CDN fallback: {url}"
            )
            http_backend = HTTPBackend()
            http_backend.download(metadata, target_dir, filename, resume_offset)


class GitBackend(DownloadBackend):
    """Downloads datasets hosted as Git repositories via safe git clone/pull commands."""

    def download(
        self, metadata: DatasetMetadata, target_dir: Path, filename: str, resume_offset: int = 0
    ) -> None:
        target_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"[GitBackend] Cloning/pulling repository {metadata.source} to {target_dir}")

        if (target_dir / ".git").exists():
            cmd = ["git", "pull"]
            cwd = target_dir
        else:
            cmd = ["git", "clone", metadata.source, str(target_dir)]
            cwd = None

        res = subprocess.run(cmd, cwd=cwd, shell=False, capture_output=True, text=True)
        if res.returncode != 0:
            raise RuntimeError(f"Git download failed: {res.stderr.strip()}")


class LocalBackend(DownloadBackend):
    """Copies or links datasets from a local file path."""

    def download(
        self, metadata: DatasetMetadata, target_dir: Path, filename: str, resume_offset: int = 0
    ) -> None:
        target_dir.mkdir(parents=True, exist_ok=True)
        src_path = Path(metadata.source)

        if src_path.is_dir():
            src_file = src_path / filename
        else:
            src_file = src_path

        if not src_file.exists():
            raise FileNotFoundError(f"[LocalBackend] Source dataset file not found: {src_file}")

        out_path = target_dir / filename
        logger.info(f"[LocalBackend] Copying {src_file} to {out_path}")
        shutil.copy2(src_file, out_path)


def get_backend(method: str) -> DownloadBackend:
    """Factory method to resolve the correct download backend."""
    backends = {
        "http": HTTPBackend,
        "https": HTTPBackend,
        "huggingface": HuggingFaceBackend,
        "git": GitBackend,
        "local": LocalBackend,
    }

    backend_cls = backends.get(method.lower())
    if not backend_cls:
        raise ValueError(f"Unsupported download method: {method}")

    return backend_cls()
