import pytest
import tempfile

from dataset.metadata.models import DatasetMetadata
from dataset.registry.registry import DatasetRegistry
from dataset.utils.exceptions import DatasetRegistrationError


@pytest.fixture
def temp_manifests_dir():
    with tempfile.TemporaryDirectory() as d:
        yield d


@pytest.fixture
def sample_metadata():
    return DatasetMetadata(
        name="test_dataset",
        version="1.0.0",
        description="Test description",
        source="http://example.com/data",
        download_method="http",
        license="MIT",
        domain="code",
        format="jsonl",
    )


def test_registry_registration(temp_manifests_dir, sample_metadata):
    registry = DatasetRegistry(manifests_dir=temp_manifests_dir)
    registry.register(sample_metadata)

    # Check retrieval
    retrieved = registry.get("test_dataset", "1.0.0")
    assert retrieved.name == "test_dataset"
    assert retrieved.domain == "code"

    # Check duplicate
    with pytest.raises(DatasetRegistrationError):
        registry.register(sample_metadata)

    # Check overwrite
    sample_metadata.description = "Updated"
    registry.register(sample_metadata, overwrite=True)
    assert registry.get("test_dataset", "1.0.0").description == "Updated"


def test_registry_list(temp_manifests_dir, sample_metadata):
    registry = DatasetRegistry(manifests_dir=temp_manifests_dir)
    sample_metadata.tags = ["nlp", "code"]
    registry.register(sample_metadata)

    ds2 = sample_metadata.model_copy(update={"name": "test2", "tags": ["vision"]})
    registry.register(ds2)

    all_ds = registry.list_datasets()
    assert len(all_ds) == 2

    code_ds = registry.list_datasets(tags=["code"])
    assert len(code_ds) == 1
    assert code_ds[0].name == "test_dataset"
