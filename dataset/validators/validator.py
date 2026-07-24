from pathlib import Path
from typing import Any

from dataset.configs.settings import config
from dataset.metadata.models import DatasetMetadata
from dataset.utils.checksums import calculate_sha256
from dataset.utils.logging import logger


class DatasetValidator:
    """
    Validates downloaded datasets against their metadata manifests.
    Checks for missing files, corrupted files, and checksum mismatches.
    """

    def __init__(self, download_dir: str | None = None):
        self.download_dir = Path(download_dir or config.download_dir)

    def validate(self, metadata: DatasetMetadata) -> dict[str, Any]:
        """
        Performs a full validation of a dataset.
        Returns a report dictionary. Raises ValidationMismatchError if strict check fails.
        """
        dataset_path = self.download_dir / metadata.name

        report = {
            "dataset": metadata.name,
            "version": metadata.version,
            "missing_files": [],
            "corrupted_files": [],
            "valid_files": [],
            "is_valid": True,
        }

        if not dataset_path.exists():
            report["is_valid"] = False
            report["missing_files"] = metadata.expected_files
            logger.error(f"Dataset directory not found: {dataset_path}")
            return report

        for filename in metadata.expected_files:
            file_path = dataset_path / filename
            if not file_path.exists():
                report["missing_files"].append(filename)
                report["is_valid"] = False
                continue

            expected_checksum = metadata.checksums.get(filename)
            if expected_checksum:
                logger.info(f"Validating checksum for {filename}...")
                actual_checksum = calculate_sha256(file_path)
                if actual_checksum != expected_checksum:
                    report["corrupted_files"].append(
                        {"file": filename, "expected": expected_checksum, "actual": actual_checksum}
                    )
                    report["is_valid"] = False
                else:
                    report["valid_files"].append(filename)
            else:
                # File exists but no checksum provided
                report["valid_files"].append(filename)

        if not report["is_valid"]:
            logger.warning(f"Validation failed for dataset {metadata.name}. Report: {report}")
        else:
            logger.info(f"Validation passed for dataset {metadata.name}.")

        return report
