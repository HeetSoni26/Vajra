from __future__ import annotations

import warnings
from model.modeling import VajraForCausalLM, VajraModel  # noqa: F401
from model.blocks import VajraBlock
from model.config import VajraConfig, ModelConfig  # noqa: F401

TransformerBlock = VajraBlock


class FoundationLM(VajraForCausalLM):
    """Deprecated alias for VajraForCausalLM."""

    def __init__(self, config: VajraConfig) -> None:
        warnings.warn(
            "FoundationLM is deprecated. Use VajraForCausalLM instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        super().__init__(config)


__all__ = ["FoundationLM", "TransformerBlock"]
