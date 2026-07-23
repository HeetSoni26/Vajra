from enum import Enum
from typing import Dict, List, Optional
from datetime import datetime
from pydantic import BaseModel, Field, ConfigDict

class DatasetStatus(str, Enum):
    REGISTERED = "registered"
    DOWNLOADING = "downloading"
    VERIFIED = "verified"
    CORRUPTED = "corrupted"
    DEPRECATED = "deprecated"

class DownloadMethod(str, Enum):
    HTTP = "http"
    HTTPS = "https"
    HUGGINGFACE = "huggingface"
    GIT = "git"
    LOCAL = "local"
    S3 = "s3"

class DatasetType(str, Enum):
    PRETRAINING = "pretraining"
    INSTRUCTION = "instruction"
    RLHF = "rlhf"
    EVALUATION = "evaluation"
    OTHER = "other"

class MaintenanceStatus(str, Enum):
    ACTIVE = "active"
    DEPRECATED = "deprecated"
    STATIC = "static"
    UNKNOWN = "unknown"

class QualityRating(str, Enum):
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    UNKNOWN = "unknown"

class DatasetMetadata(BaseModel):
    """
    Standardized schema for dataset metadata definition.
    """
    # Identity
    name: str = Field(..., description="Unique dataset name.")
    version: str = Field(..., description="Semantic version of the dataset.")
    description: str = Field(..., description="Detailed description of the dataset contents.")
    
    # Sourcing
    source: str = Field(..., description="Source URL, URI, or Hugging Face repo ID.")
    homepage: Optional[str] = Field(default=None, description="Project or dataset homepage.")
    download_method: DownloadMethod = Field(..., description="The backend required to download the dataset.")
    
    # Categorization
    license: str = Field(..., description="License type (e.g., Apache 2.0, MIT, ODC-BY).")
    language: str = Field(default="en", description="Primary language of the dataset (e.g., 'en', 'multi').")
    domain: str = Field(..., description="Data domain (e.g., 'code', 'web', 'math', 'medical').")
    format: str = Field(..., description="Format of raw files (e.g., 'jsonl', 'parquet', 'txt').")
    tags: List[str] = Field(default_factory=list, description="List of categorical tags.")
    dataset_type: DatasetType = Field(default=DatasetType.OTHER, description="Type of dataset (pretraining, instruction, etc).")
    
    # Verification & Sizing
    expected_files: List[str] = Field(default_factory=list, description="List of expected filenames upon completion.")
    estimated_size_bytes: int = Field(default=0, description="Estimated total download size in bytes.")
    checksums: Dict[str, str] = Field(
        default_factory=dict,
        description="Mapping of filename to SHA256 checksum for integrity verification."
    )
    num_documents: Optional[int] = Field(default=None, description="Number of documents if known.")
    estimated_tokens: Optional[int] = Field(default=None, description="Estimated tokens if known.")
    
    # Quality & Lifecycle
    quality_rating: QualityRating = Field(default=QualityRating.UNKNOWN, description="Quality rating.")
    maintenance_status: MaintenanceStatus = Field(default=MaintenanceStatus.UNKNOWN, description="Maintenance activity.")
    recommended_use: Optional[str] = Field(default=None, description="Recommended usage notes.")
    notes: Optional[str] = Field(default=None, description="Additional context or notes.")
    
    # Academic/Tracking
    citation: Optional[str] = Field(default=None, description="BibTeX or citation string.")
    status: DatasetStatus = Field(default=DatasetStatus.REGISTERED, description="Current tracking status.")
    last_updated: datetime = Field(default_factory=datetime.utcnow, description="Last update timestamp.")
    
    model_config = ConfigDict(
        use_enum_values=True,
        json_schema_extra={
            "example": {
                "name": "sample_code_dataset",
                "version": "1.0.0",
                "description": "A sample dataset of open source code.",
                "source": "https://example.com/data.jsonl",
                "download_method": "https",
                "license": "Apache 2.0",
                "domain": "code",
                "format": "jsonl",
                "dataset_type": "pretraining"
            }
        }
    )
