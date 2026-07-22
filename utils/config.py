from __future__ import annotations

from pathlib import Path
from typing import Any

from .file_utils import read_yaml


def merge_configs(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    """Recursively merge override dictionary into base dictionary."""
    merged = base.copy()
    for key, value in override.items():
        if key in merged and isinstance(merged[key], dict) and isinstance(value, dict):
            merged[key] = merge_configs(merged[key], value)
        else:
            merged[key] = value
    return merged


def apply_cli_overrides(config: dict[str, Any], overrides: list[str]) -> dict[str, Any]:
    """Apply dot-notation key=value CLI overrides (e.g., ['model.hidden_size=512', 'learning_rate=1e-4'])."""
    result = config.copy()
    for item in overrides:
        if "=" not in item:
            continue
        key_path, raw_val = item.split("=", 1)
        keys = key_path.strip().split(".")
        
        # Parse scalar types (int, float, bool, str)
        val: Any = raw_val.strip()
        if val.lower() == "true":
            val = True
        elif val.lower() == "false":
            val = False
        else:
            try:
                val = int(val)
            except ValueError:
                try:
                    val = float(val)
                except ValueError:
                    pass

        curr = result
        for k in keys[:-1]:
            curr = curr.setdefault(k, {})
        curr[keys[-1]] = val
    return result


def load_config(path: str | Path, cli_overrides: list[str] | None = None) -> dict[str, Any]:
    """Load configuration from YAML file and apply CLI overrides if provided."""
    config = read_yaml(path)
    if cli_overrides:
        config = apply_cli_overrides(config, cli_overrides)
    return config
