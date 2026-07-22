from __future__ import annotations

import torch
from torch import nn
import torch.nn.functional as F
import torch.utils.checkpoint

from .attention import GQAAttention
from .config import ModelConfig
from .feedforward import SwiGLU
from .norm import RMSNorm
from .rope import precompute_rope_frequencies


class TransformerBlock(nn.Module):
    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.input_layernorm = RMSNorm(config.hidden_size, config.rms_norm_eps)
        self.self_attn = GQAAttention(config)
        self.post_attention_layernorm = RMSNorm(config.hidden_size, config.rms_norm_eps)
        self.mlp = SwiGLU(config.hidden_size, config.intermediate_size)

    def forward(
        self,
        x: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        kv_cache: object | None = None,
        layer_idx: int = 0,
    ) -> torch.Tensor:
        x = x + self.self_attn(self.input_layernorm(x), cos, sin, kv_cache=kv_cache, layer_idx=layer_idx)
        x = x + self.mlp(self.post_attention_layernorm(x))
        return x


class FoundationLM(nn.Module):
    """Decoder-only LLaMA-style foundation model."""

    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.config = config
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        self.layers = nn.ModuleList([TransformerBlock(config) for _ in range(config.num_layers)])
        self.norm = RMSNorm(config.hidden_size, config.rms_norm_eps)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        if config.tie_word_embeddings:
            self.lm_head.weight = self.embed_tokens.weight

        cos, sin = precompute_rope_frequencies(
            config.head_dim, config.max_position_embeddings, config.rope_theta
        )
        self.register_buffer("rope_cos", cos, persistent=False)
        self.register_buffer("rope_sin", sin, persistent=False)
        self.use_gradient_checkpointing = getattr(config, "use_gradient_checkpointing", False)
        self.apply(self._init_weights)

    def _init_weights(self, module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(
        self,
        input_ids: torch.Tensor,
        labels: torch.Tensor | None = None,
        kv_cache: object | None = None,
        start_pos: int = 0,
    ) -> dict[str, torch.Tensor]:
        seq_len = input_ids.shape[1]
        x = self.embed_tokens(input_ids)

        # Slice RoPE frequencies for the correct position range
        cos = self.rope_cos[start_pos: start_pos + seq_len]
        sin = self.rope_sin[start_pos: start_pos + seq_len]

        for layer_idx, layer in enumerate(self.layers):
            if self.use_gradient_checkpointing and self.training:
                x = torch.utils.checkpoint.checkpoint(
                    layer, x, cos, sin, None, layer_idx, use_reentrant=False
                )
            else:
                x = layer(x, cos, sin, kv_cache=kv_cache, layer_idx=layer_idx)

        logits = self.lm_head(self.norm(x))
        out: dict[str, torch.Tensor] = {"logits": logits}
        if labels is not None:
            loss = F.cross_entropy(logits[:, :-1].reshape(-1, logits.size(-1)), labels[:, 1:].reshape(-1))
            out["loss"] = loss
        return out
