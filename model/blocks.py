import torch
from torch import nn

from model.config import VajraConfig
from model.layers.attention import VajraAttention
from model.layers.mlp import VajraMLP
from model.layers.rmsnorm import RMSNorm


class VajraBlock(nn.Module):
    def __init__(self, config: VajraConfig):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.input_layernorm = RMSNorm(config.hidden_size, eps=config.rmsnorm_eps)
        self.self_attn = VajraAttention(config=config)
        self.post_attention_layernorm = RMSNorm(config.hidden_size, eps=config.rmsnorm_eps)
        self.mlp = VajraMLP(config)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.LongTensor | None = None,
        past_key_value: tuple[torch.Tensor, torch.Tensor] | None = None,
        output_attentions: bool | None = False,
        use_cache: bool | None = False,
        cos: torch.Tensor | None = None,
        sin: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor] | None]:
        residual = hidden_states

        hidden_states = self.input_layernorm(hidden_states)

        # Self Attention
        hidden_states, self_attn_weights, present_key_value = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=past_key_value,
            output_attentions=output_attentions,
            use_cache=use_cache,
            cos=cos,
            sin=sin,
        )
        hidden_states = residual + hidden_states

        # Fully Connected
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states

        outputs = (hidden_states,)

        if output_attentions:
            outputs += (self_attn_weights,)

        if use_cache:
            outputs += (present_key_value,)

        return outputs
