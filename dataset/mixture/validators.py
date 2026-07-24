from dataset.metadata.licenses import LicenseCategory, LicenseValidator
from dataset.mixture.models import DatasetMixture


class MixtureValidationError(Exception):
    pass


class MixtureValidator:
    """
    Validates a DatasetMixture against standard rules.
    """

    @classmethod
    def validate(cls, mixture: DatasetMixture) -> dict[str, list[str]]:
        """
        Validates the mixture and returns a dictionary of warnings and errors.
        """
        errors = []
        warnings = []

        if not mixture.entries:
            errors.append("Mixture has no entries.")
            return {"errors": errors, "warnings": warnings}

        enabled_entries = [e for e in mixture.entries if e.is_enabled]

        if not enabled_entries:
            errors.append("Mixture has no enabled entries.")
            return {"errors": errors, "warnings": warnings}

        # 1. Weights sum to 100%
        total_weight = sum(e.weight for e in enabled_entries)
        if abs(total_weight - 100.0) > 0.01:
            errors.append(
                f"Total weight of enabled entries is {total_weight}%, but must sum to 100%."
            )

        # 2. No duplicate entries
        seen_names = set()
        for e in enabled_entries:
            key = f"{e.name}:{e.version}"
            if key in seen_names:
                errors.append(f"Duplicate entry found for dataset: {key}")
            seen_names.add(key)

        # 3. Valid licenses
        for e in enabled_entries:
            category = LicenseValidator.classify(e.license)
            if category == LicenseCategory.RESTRICTED:
                errors.append(f"Dataset {e.name} has a restricted license: {e.license}")
            elif category == LicenseCategory.UNKNOWN:
                warnings.append(f"Dataset {e.name} has an unknown license: {e.license}")

        # 4. Missing metadata & Invalid Estimates
        for e in enabled_entries:
            if e.estimated_tokens is None or e.estimated_tokens <= 0:
                warnings.append(f"Dataset {e.name} has invalid or missing estimated_tokens.")
            if e.estimated_documents is None or e.estimated_documents <= 0:
                warnings.append(f"Dataset {e.name} has invalid or missing estimated_documents.")

        return {"errors": errors, "warnings": warnings}
