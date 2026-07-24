"""Custom exceptions for the Dataset Collection Framework."""


class DatasetError(Exception):
    """Base class for all dataset-related errors."""


class DatasetRegistrationError(DatasetError):
    """Raised when there is an issue registering or loading a dataset."""


class DownloadFailedError(DatasetError):
    """Raised when a dataset download fails (after retries)."""


class ValidationMismatchError(DatasetError):
    """Raised when a dataset file fails checksum or format validation."""


class ConfigurationError(DatasetError):
    """Raised when the dataset framework configuration is invalid."""
