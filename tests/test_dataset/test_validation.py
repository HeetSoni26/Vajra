import pytest
import tempfile
import hashlib
from pathlib import Path

from dataset.metadata.models import DatasetMetadata
from dataset.validators.validator import DatasetValidator


@pytest.fixture
def temp_download_dir():
    with tempfile.TemporaryDirectory() as d:
        yield d


@pytest.fixture
def test_dataset_setup(temp_download_dir):
    ds_dir = Path(temp_download_dir) / "test_ds"
    ds_dir.mkdir()

    file1 = ds_dir / "data1.txt"
    file1.write_text("hello world")

    # Calculate sha256 for "hello world"
    sha = hashlib.sha256(b"hello world").hexdigest()

    metadata = DatasetMetadata(
        name="test_ds",
        version="1.0.0",
        description="test",
        source="local",
        download_method="local",
        license="MIT",
        domain="test",
        format="txt",
        expected_files=["data1.txt", "missing.txt"],
        checksums={"data1.txt": sha},
    )
    return metadata


def test_validator_missing_and_valid(temp_download_dir, test_dataset_setup):
    validator = DatasetValidator(download_dir=temp_download_dir)
    report = validator.validate(test_dataset_setup)

    assert report["is_valid"] is False
    assert "data1.txt" in report["valid_files"]
    assert "missing.txt" in report["missing_files"]


def test_validator_corrupted(temp_download_dir, test_dataset_setup):
    # Corrupt the file
    file1 = Path(temp_download_dir) / "test_ds" / "data1.txt"
    file1.write_text("corrupted data")

    validator = DatasetValidator(download_dir=temp_download_dir)
    report = validator.validate(test_dataset_setup)

    assert report["is_valid"] is False
    assert len(report["corrupted_files"]) == 1
    assert report["corrupted_files"][0]["file"] == "data1.txt"
