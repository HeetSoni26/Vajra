import time
from pathlib import Path

from dataset.configs.settings import config
from dataset.metadata.models import DatasetMetadata
from dataset.cache.manager import CacheManager
from dataset.downloaders.backends import get_backend
from dataset.utils.logging import logger
from dataset.utils.exceptions import DownloadFailedError

class DownloadManager:
    """
    Orchestrates the downloading of datasets using configured backends.
    Handles retries, cache state resumption, and concurrency logic.
    """

    def __init__(self, download_dir: str | None = None):
        self.download_dir = Path(download_dir or config.download_dir)
        self.cache_manager = CacheManager()

    def download_dataset(self, metadata: DatasetMetadata) -> None:
        """
        Downloads all expected files for a dataset.
        """
        target_dir = self.download_dir / metadata.name
        target_dir.mkdir(parents=True, exist_ok=True)
        
        backend = get_backend(metadata.download_method)
        logger.info(f"Starting download for dataset: {metadata.name} (v{metadata.version})")
        
        for filename in metadata.expected_files:
            if self.cache_manager.is_completed(metadata.name, filename):
                logger.info(f"Skipping {filename}: already fully downloaded.")
                continue

            resume_offset = self.cache_manager.get_downloaded_bytes(metadata.name, filename)
            
            # Retry loop
            success = False
            for attempt in range(1, config.retry_count + 1):
                try:
                    logger.info(f"Downloading {filename} (Attempt {attempt}/{config.retry_count})")
                    backend.download(metadata, target_dir, filename, resume_offset)
                    
                    # If we reach here without exception, simulate success marking
                    self.cache_manager.update_progress(metadata.name, filename, metadata.estimated_size_bytes, completed=True)
                    success = True
                    break
                    
                except Exception as e:
                    logger.warning(f"Download failed for {filename} on attempt {attempt}: {e}")
                    time.sleep(2 ** attempt) # Exponential backoff
            
            if not success:
                logger.error(f"Failed to download {filename} after {config.retry_count} attempts.")
                raise DownloadFailedError(f"Could not download {filename}")

        logger.info(f"Dataset {metadata.name} download complete.")
