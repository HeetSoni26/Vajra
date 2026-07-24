from typing import List, Optional
from pydantic import BaseModel, Field
from datetime import datetime
from dataset.metadata.models import QualityRating


class SamplingStrategy(BaseModel):
    """
    Abstract interface placeholder for sampling strategies.
    """

    type: str = Field(
        ..., description="The sampling strategy type (e.g., fixed, temperature, dynamic)."
    )


class DatasetMixtureEntry(BaseModel):
    """
    An individual dataset entry within a mixture.
    """

    name: str = Field(..., description="Dataset name.")
    version: str = Field(default="1.0.0", description="Dataset version.")
    weight: float = Field(..., description="Percentage weight (0.0 to 100.0) within the mixture.")
    sampling_strategy: Optional[SamplingStrategy] = Field(
        default=None, description="Sampling strategy interface."
    )

    # Cached metadata from registry for quick analysis without lookup
    language: str = Field(default="en")
    domain: str = Field(default="mixed")
    priority: int = Field(default=1, description="Priority scale (e.g., 1=highest).")
    license: str = Field(default="unknown")

    estimated_tokens: Optional[int] = Field(default=None)
    estimated_documents: Optional[int] = Field(default=None)
    quality_rating: QualityRating = Field(default=QualityRating.UNKNOWN)

    notes: Optional[str] = Field(default=None)
    is_enabled: bool = Field(
        default=True, description="Whether this dataset is actively included in the mixture."
    )


class DatasetMixture(BaseModel):
    """
    A collection of dataset entries forming a training corpus mixture.
    """

    name: str = Field(..., description="Unique name of the mixture.")
    description: str = Field(default="", description="Description of the mixture's purpose.")
    version: str = Field(default="1.0", description="Version of the mixture.")
    created_at: datetime = Field(default_factory=datetime.utcnow)

    entries: List[DatasetMixtureEntry] = Field(default_factory=list)
