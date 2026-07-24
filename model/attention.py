from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

from .config import ModelConfig
from .rope import apply_rope


class GQAAttention(nn.Module):
    """Grouped-query causal self-attention with optional FlashAttention fallback."""

    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.config = config
        self.num_heads = config.num_attention_heads
        self.num_kv_heads = config.num_key_value_heads
        self.head_dim = config.head_dim
        self.num_kv_groups = self.num_heads // self.num_kv_heads
        if self.num_heads % self.num_kv_heads != 0:
            raise ValueError("num_attention_heads must be divisible by num_key_value_heads")

        self.q_proj = nn.Linear(
            config.hidden_size, config.num_attention_heads * self.head_dim, bias=False
        )
        self.k_proj = nn.Linear(
            config.hidden_size, config.num_key_value_heads * self.head_dim, bias=False
        )
        self.v_proj = nn.Linear(
            config.hidden_size, config.num_key_value_heads * self.head_dim, bias=False
        )
        self.o_proj = nn.Linear(
            config.num_attention_heads * self.head_dim, config.hidden_size, bias=False
        )

    def forward(
        self,
        x: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        kv_cache: object | None = None,
        layer_idx: int = 0,
    ) -> torch.Tensor:
        batch, seq_len, _ = x.shape
        q = self.q_proj(x).view(batch, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(batch, seq_len, self.num_kv_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(batch, seq_len, self.num_kv_heads, self.head_dim).transpose(1, 2)

        q = apply_rope(q, cos, sin)
        k = apply_rope(k, cos, sin)

        # KV Cache path — append and retrieve full key/value history
        if kv_cache is not None:
            k, v = kv_cache.update(layer_idx, k, v)

        k = k.repeat_interleave(self.num_kv_groups, dim=1)
        v = v.repeat_interleave(self.num_kv_groups, dim=1)

        # Use is_causal=True only during prefill (seq_len > 1);
        # during decode (seq_len == 1) the single query attends to all keys.
        is_causal = (q.shape[2] == k.shape[2]) and (seq_len > 1)
        attn = F.scaled_dot_product_attention(q, k, v, is_causal=is_causal)
        attn = attn.transpose(1, 2).contiguous().view(batch, seq_len, -1)
        return self.o_proj(attn)
