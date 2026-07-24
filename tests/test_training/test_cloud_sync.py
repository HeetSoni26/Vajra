import shutil

import pytest
import yaml

from training.cloud.sync_manager import CloudSyncManager
from training.resume import ResumeManager
from tests.test_training.test_resume import create_dummy_checkpoint


@pytest.fixture
def sync_env(tmp_path):
    local_dir = tmp_path / "local_checkpoints"
    local_dir.mkdir()

    remote_dir = tmp_path / "remote_checkpoints"
    remote_dir.mkdir()

    config_dir = tmp_path / "configs"
    config_dir.mkdir()
    config_path = config_dir / "cloud_sync.yaml"

    config_data = {
        "enable_sync": True,
        "provider": "local",  # Use local backend for testing
        "background_upload": False,  # Sync immediately for testing
        "retry_limit": 3,
        "retry_interval": 0,
        "download_on_resume": True,
    }
    config_path.write_text(yaml.dump(config_data))

    # Overwrite the LocalBackend's target_dir globally for this test
    from training.cloud.backends import LocalBackend

    original_init = LocalBackend.__init__

    def mock_init(self, target_dir):
        original_init(self, remote_dir)

    LocalBackend.__init__ = mock_init

    yield local_dir, remote_dir, config_path

    LocalBackend.__init__ = original_init


def test_background_upload_and_download(sync_env):
    local_dir, remote_dir, config_path = sync_env

    manager = CloudSyncManager(config_path)
    assert manager.enabled is True

    # Create a local experiment
    exp1 = local_dir / "exp_1"
    exp1.mkdir()
    create_dummy_checkpoint(exp1, step=10)

    # Sync it to "cloud"
    manager.sync_experiment(exp1)

    # Verify it exists in remote
    remote_exp1 = remote_dir / "experiments" / "exp_1"
    assert remote_exp1.exists()
    assert (remote_exp1 / "latest.pt").exists()

    # Delete local
    shutil.rmtree(exp1)
    assert not exp1.exists()

    # Test remote discovery
    remote_exps = manager.discover_remote_experiments()
    assert len(remote_exps) == 1
    assert remote_exps[0] == "exp_1"

    # Test download
    dl_dir = manager.download_experiment("exp_1", local_dir)
    assert dl_dir.exists()
    assert (dl_dir / "latest.pt").exists()


def test_resume_manager_with_cloud_fallback(sync_env):
    local_dir, remote_dir, config_path = sync_env

    # Create remote experiment manually
    remote_exp = remote_dir / "experiments" / "exp_100"
    remote_exp.mkdir(parents=True)
    create_dummy_checkpoint(remote_exp, step=100)

    # Instantiate ResumeManager (which initializes its own CloudSyncManager, but we need to patch its config path)
    rm = ResumeManager(local_dir)
    rm.cloud_sync = CloudSyncManager(config_path)

    # Local dir is empty. ResumeManager should fallback to cloud, download exp_100, and return it.
    exp_dir, state = rm.find_latest_valid_experiment(prefix="exp_")

    assert exp_dir.name == "exp_100"
    assert exp_dir.exists()
    assert state["step"] == 100


def test_cloud_sync_retry_logic(sync_env, monkeypatch):
    local_dir, remote_dir, config_path = sync_env

    manager = CloudSyncManager(config_path)

    # Mock the backend upload_folder to fail twice then succeed
    call_count = 0

    def mock_upload(*args, **kwargs):
        nonlocal call_count
        call_count += 1
        if call_count < 3:
            raise ConnectionError("Simulated network drop")
        # Let it succeed on the 3rd try (we won't actually copy, just pass)
        pass

    monkeypatch.setattr(manager.backend, "upload_folder", mock_upload)

    exp1 = local_dir / "exp_1"
    exp1.mkdir()

    manager.sync_experiment(exp1)
    assert call_count == 3
