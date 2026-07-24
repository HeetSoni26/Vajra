from pathlib import Path
from dataset.sharding.reader import BinaryShardReader


class ShardValidator:
    """
    Validates physical binary shards against their configuration constraints.
    """

    @staticmethod
    def validate_shard(metadata_path: Path) -> dict:
        """
        Runs comprehensive consistency, sequence length, and token range checks.
        """
        results = {"valid": True, "errors": []}

        try:
            reader = BinaryShardReader(metadata_path)

            # 1. Integrity
            if not reader.verify_integrity():
                results["valid"] = False
                results["errors"].append("Checksum validation failed. Corrupted binary.")
                return results

            # 2. Sequence consistency
            for idx, seq in enumerate(reader.stream()):
                if len(seq) != reader.metadata.sequence_length:
                    results["valid"] = False
                    results["errors"].append(
                        f"Sequence {idx} length mismatch. Expected {reader.metadata.sequence_length}, got {len(seq)}."
                    )

                # 3. Token range validation
                max_token = seq.max() if len(seq) > 0 else 0
                if max_token >= reader.metadata.vocab_size:
                    results["valid"] = False
                    results["errors"].append(
                        f"Sequence {idx} contains out-of-bounds token {max_token} for vocab size {reader.metadata.vocab_size}."
                    )

        except Exception as e:
            results["valid"] = False
            results["errors"].append(f"Validation crashed: {str(e)}")

        return results
