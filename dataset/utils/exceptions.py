"""Custom exceptions for the Dataset Collection Framework."""

class DatasetError(Exception):
    """Base class for all dataset-related errors."""
    pass

class DatasetRegistrationError(DatasetError):
    """Raised when there is an issue registering or loading a dataset."""
    pass

class DownloadFailedError(DatasetError):
    """Raised when a dataset download fails (after retries)."""
    pass

class ValidationMismatchError(DatasetError):
    """Raised when a dataset file fails checksum or format validation."""
    pass

class ConfigurationError(DatasetError):
    """Raised when the dataset framework configuration is invalid."""
    pass
