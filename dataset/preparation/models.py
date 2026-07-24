from typing import Any

from pydantic import BaseModel, Field


class Document(BaseModel):
    """
    Standardized internal representation of a dataset document.
    """

    id: str = Field(..., description="Unique document identifier.")
    text: str = Field(..., description="The main text content.")
    metadata: dict[str, Any] = Field(
        default_factory=dict, description="Preserved metadata (e.g. url, date)."
    )


class PreparationConfig(BaseModel):
    """
    Configuration for the preparation pipeline.
    Controls which stages are enabled and sets thresholds.
    """

    # Stages toggles
    enable_unicode_normalization: bool = True
    enable_whitespace_normalization: bool = True
    enable_empty_removal: bool = True
    enable_length_filtering: bool = True
    enable_exact_deduplication: bool = True
    enable_language_detection: bool = False

    # Thresholds
    min_length: int = 50
    max_length: int = 1000000
    min_char_ratio: float = 0.2
    max_whitespace_ratio: float = 0.5

    # Execution
    max_workers: int = 4
    batch_size: int = 1000


class PreparationStatistics(BaseModel):
    """
    Tracks statistics during pipeline execution.
    """

    total_documents_read: int = 0
    total_documents_written: int = 0
    total_characters_read: int = 0
    total_characters_written: int = 0

    filtered_empty: int = 0
    filtered_length: int = 0
    filtered_duplicates: int = 0
    filtered_quality: int = 0

    processing_time_seconds: float = 0.0

    @property
    def average_length_written(self) -> float:
        if self.total_documents_written == 0:
            return 0.0
        return self.total_characters_written / self.total_documents_written
