import platform
import sys
import subprocess
from typing import Dict, Any


class SnapshotManager:
    """
    Captures system, environment, and configuration snapshots for reproducibility.
    """

    @staticmethod
    def capture_system_snapshot() -> Dict[str, Any]:
        snapshot = {
            "os": platform.system(),
            "os_release": platform.release(),
            "python_version": sys.version,
        }

        # PyTorch & CUDA
        try:
            import torch

            snapshot["pytorch_version"] = torch.__version__
            snapshot["cuda_available"] = torch.cuda.is_available()
            if torch.cuda.is_available():
                snapshot["cuda_version"] = torch.version.cuda
        except ImportError:
            snapshot["pytorch_version"] = None

        # Git commit
        try:
            commit = (
                subprocess.check_output(["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL)
                .decode("utf-8")
                .strip()
            )
            snapshot["git_commit"] = commit
        except Exception:
            snapshot["git_commit"] = None

        return snapshot

    @staticmethod
    def capture_config_snapshot(
        training_config: Dict[str, Any] = None,
        model_config: Dict[str, Any] = None,
        dataset_mixture: Dict[str, Any] = None,
    ) -> Dict[str, Any]:
        return {
            "training_config": training_config or {},
            "model_config": model_config or {},
            "dataset_mixture": dataset_mixture or {},
        }
