from enum import Enum
from typing import Dict

class LicenseCategory(str, Enum):
    COMMERCIALLY_USABLE = "commercially_usable"
    RESEARCH_ONLY = "research_only"
    RESTRICTED = "restricted"
    UNKNOWN = "unknown"

LICENSE_MAP: Dict[str, LicenseCategory] = {
    "apache 2.0": LicenseCategory.COMMERCIALLY_USABLE,
    "mit": LicenseCategory.COMMERCIALLY_USABLE,
    "bsd": LicenseCategory.COMMERCIALLY_USABLE,
    "bsd-2-clause": LicenseCategory.COMMERCIALLY_USABLE,
    "bsd-3-clause": LicenseCategory.COMMERCIALLY_USABLE,
    "cc-by": LicenseCategory.COMMERCIALLY_USABLE,
    "cc-by-sa": LicenseCategory.COMMERCIALLY_USABLE,
    "cc0": LicenseCategory.COMMERCIALLY_USABLE,
    "public domain": LicenseCategory.COMMERCIALLY_USABLE,
    "odc": LicenseCategory.COMMERCIALLY_USABLE,
    "odc-by": LicenseCategory.COMMERCIALLY_USABLE,
    
    "research only": LicenseCategory.RESEARCH_ONLY,
    "cc-by-nc": LicenseCategory.RESEARCH_ONLY,
    "cc-by-nc-sa": LicenseCategory.RESEARCH_ONLY,
    
    "restricted": LicenseCategory.RESTRICTED,
    "custom": LicenseCategory.UNKNOWN,
}

class LicenseValidator:
    """
    Validates and classifies dataset licenses.
    """
    @classmethod
    def classify(cls, license_name: str) -> LicenseCategory:
        """
        Classify a given license string into a LicenseCategory.
        """
        key = license_name.strip().lower()
        return LICENSE_MAP.get(key, LicenseCategory.UNKNOWN)
