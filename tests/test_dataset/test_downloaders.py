import tempfile
from unittest.mock import MagicMock

import pytest

from dataset.cache.manager import CacheManager
from dataset.downloaders.manager import DownloadManager
from dataset.metadata.models import DatasetMetadata
from dataset.utils.exceptions import DownloadFailedError


@pytest.fixture
def temp_download_dir():
    with tempfile.TemporaryDirectory() as d:
        yield d


@pytest.fixture
def sample_metadata():
    return DatasetMetadata(
        name="test_dl",
        version="1.0.0",
        description="test",
        source="local",
        download_method="local",
        license="MIT",
        domain="test",
        format="txt",
        expected_files=["data.txt"],
        estimated_size_bytes=100,
    )


def test_download_manager_success(temp_download_dir, sample_metadata, monkeypatch):
    # Mock backend factory
    mock_backend = MagicMock()
    monkeypatch.setattr("dataset.downloaders.manager.get_backend", lambda x: mock_backend)

    # Configure config via monkeypatch to use temp dir
    monkeypatch.setattr("dataset.configs.settings.config.download_dir", temp_download_dir)
    monkeypatch.setattr("dataset.configs.settings.config.cache_dir", temp_download_dir)

    manager = DownloadManager(download_dir=temp_download_dir)
    manager.cache_manager = CacheManager(cache_dir=temp_download_dir)

    manager.download_dataset(sample_metadata)

    mock_backend.download.assert_called_once()
    assert manager.cache_manager.is_completed("test_dl", "data.txt")


def test_download_manager_failure(temp_download_dir, sample_metadata, monkeypatch):
    # Mock backend factory to fail
    mock_backend = MagicMock()
    mock_backend.download.side_effect = Exception("Network error")
    monkeypatch.setattr("dataset.downloaders.manager.get_backend", lambda x: mock_backend)

    # Reduce retries for fast test
    monkeypatch.setattr("dataset.configs.settings.config.retry_count", 2)

    manager = DownloadManager(download_dir=temp_download_dir)
    manager.cache_manager = CacheManager(cache_dir=temp_download_dir)

    with pytest.raises(DownloadFailedError):
        manager.download_dataset(sample_metadata)

    assert mock_backend.download.call_count == 2
