import os
from typing import Optional
from pydantic_settings import BaseSettings
from pydantic import Field

class DatasetConfig(BaseSettings):
    """
    Global configuration for the Vajra Dataset Collection Framework.
    Values can be overridden by environment variables prefixed with VAJRA_DATASET_.
    """
    # Directories
    download_dir: str = Field(
        default=os.path.join(os.getcwd(), "data", "raw"),
        description="Target directory for downloaded raw datasets."
    )
    cache_dir: str = Field(
        default=os.path.join(os.getcwd(), ".cache", "vajra_datasets"),
        description="Directory for resumable download states and caching."
    )
    manifests_dir: str = Field(
        default=os.path.join(os.getcwd(), "dataset", "manifests"),
        description="Directory storing dataset registration manifests."
    )
    
    # Network and Performance
    max_workers: int = Field(
        default=4,
        description="Maximum concurrent download workers."
    )
    timeout_seconds: int = Field(
        default=300,
        description="Timeout for network requests."
    )
    retry_count: int = Field(
        default=3,
        description="Number of retries for failed downloads."
    )
    
    # Proxies
    http_proxy: Optional[str] = Field(
        default=None,
        description="Optional HTTP proxy url."
    )
    https_proxy: Optional[str] = Field(
        default=None,
        description="Optional HTTPS proxy url."
    )
    
    # Logging
    log_level: str = Field(
        default="INFO",
        description="Logging verbosity (DEBUG, INFO, WARNING, ERROR)."
    )

    class Config:
        env_prefix = "VAJRA_DATASET_"
        env_file = ".env"
        extra = "ignore"

# Global configuration instance
config = DatasetConfig()
