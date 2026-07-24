from dataset.configs.settings import DatasetConfig


def test_config_defaults():
    config = DatasetConfig()
    assert config.max_workers == 4
    assert config.retry_count == 3
    assert config.log_level == "INFO"
    assert config.http_proxy is None


def test_config_env_override(monkeypatch):
    monkeypatch.setenv("VAJRA_DATASET_MAX_WORKERS", "10")
    monkeypatch.setenv("VAJRA_DATASET_LOG_LEVEL", "DEBUG")

    config = DatasetConfig()
    assert config.max_workers == 10
    assert config.log_level == "DEBUG"
