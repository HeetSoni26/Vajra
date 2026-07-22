"""KV Cache for autoregressive generation with the FoundationLM decoder."""

from __future__ import annotations

from dataclasses import dataclass, field

import torch


@dataclass
class KVCache:
    """Per-layer key-value cache for efficient autoregressive decoding.

    Supports both prefill (processing full prompt) and decode (single-token
    generation) modes with dynamic sequence length tracking.
    """

    num_layers: int
    max_batch_size: int = 1
    max_seq_len: int = 4096

    # Internal storage — populated lazily on first update
    _keys: list[torch.Tensor | None] = field(default_factory=list, repr=False)
    _values: list[torch.Tensor | None] = field(default_factory=list, repr=False)
    _seq_len: int = field(default=0, repr=False)

    def __post_init__(self) -> None:
        self._keys = [None] * self.num_layers
        self._values = [None] * self.num_layers
        self._seq_len = 0

    @property
    def seq_len(self) -> int:
        """Current cached sequence length."""
        return self._seq_len

    def update(
        self,
        layer_idx: int,
        key: torch.Tensor,
        value: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Append new key/value tensors for a layer and return the full cached k/v.

        Args:
            layer_idx: Index of the transformer layer.
            key: New key tensor of shape ``[batch, num_kv_heads, new_seq, head_dim]``.
            value: New value tensor of shape ``[batch, num_kv_heads, new_seq, head_dim]``.

        Returns:
            Tuple of ``(cached_keys, cached_values)`` containing the full
            concatenated history for this layer.
        """
        if self._keys[layer_idx] is None:
            self._keys[layer_idx] = key
            self._values[layer_idx] = value
        else:
            self._keys[layer_idx] = torch.cat([self._keys[layer_idx], key], dim=2)
            self._values[layer_idx] = torch.cat([self._values[layer_idx], value], dim=2)

        # Track sequence length from layer 0
        if layer_idx == 0:
            self._seq_len = self._keys[0].shape[2]

        return self._keys[layer_idx], self._values[layer_idx]

    def reset(self) -> None:
        """Clear all cached key/value tensors and free GPU memory."""
        for i in range(self.num_layers):
            self._keys[i] = None
            self._values[i] = None
        self._seq_len = 0
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def get(self, layer_idx: int) -> tuple[torch.Tensor | None, torch.Tensor | None]:
        """Get cached key/value for a specific layer."""
        return self._keys[layer_idx], self._values[layer_idx]
