import torch
import torch.nn as nn
from typing import Optional, Tuple, List, Union
from model.config import VajraConfig
from model.layers.rmsnorm import RMSNorm
from model.layers.rope import RotaryEmbedding
from model.blocks import VajraBlock

class VajraModel(nn.Module):
    def __init__(self, config: VajraConfig):
        super().__init__()
        self.config = config
        self.vocab_size = config.vocab_size
        
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        
        self.layers = nn.ModuleList([VajraBlock(config) for _ in range(config.num_layers)])
        self.norm = RMSNorm(config.hidden_size, eps=config.rmsnorm_eps)
        self.rotary_emb = RotaryEmbedding(
            dim=config.hidden_size // config.num_attention_heads,
            max_position_embeddings=config.max_position_embeddings,
            base=config.rope_theta,
        )

    def forward(
        self,
        input_ids: torch.LongTensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[Tuple[torch.Tensor, torch.Tensor]]] = None,
        use_cache: Optional[bool] = None,
    ) -> Tuple[torch.Tensor, Optional[List[Tuple[torch.Tensor, torch.Tensor]]]]:
        bsz, seq_len = input_ids.shape

        if position_ids is None:
            device = input_ids.device
            past_key_values_length = 0
            if past_key_values is not None:
                past_key_values_length = past_key_values[0][0].shape[1]
            position_ids = torch.arange(
                past_key_values_length, seq_len + past_key_values_length, dtype=torch.long, device=device
            )
            position_ids = position_ids.unsqueeze(0).view(-1, seq_len)

        hidden_states = self.embed_tokens(input_ids)
        
        # Prepare RoPE
        kv_seq_len = hidden_states.shape[1]
        if past_key_values is not None:
            kv_seq_len += past_key_values[0][0].shape[1]
            
        cos, sin = self.rotary_emb(hidden_states, seq_len=kv_seq_len)
        
        # Expand attention mask if provided
        if attention_mask is not None:
            # [bsz, seq_len] -> [bsz, 1, seq_len, kv_seq_len]
            attention_mask = self._prepare_decoder_attention_mask(
                attention_mask, (bsz, seq_len), hidden_states, kv_seq_len
            )

        next_decoder_cache = () if use_cache else None

        for idx, decoder_layer in enumerate(self.layers):
            past_key_value = past_key_values[idx] if past_key_values is not None else None

            layer_outputs = decoder_layer(
                hidden_states,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_value=past_key_value,
                output_attentions=False,
                use_cache=use_cache,
                cos=cos,
                sin=sin,
            )

            hidden_states = layer_outputs[0]

            if use_cache:
                next_decoder_cache += (layer_outputs[-1],)

        hidden_states = self.norm(hidden_states)

        return hidden_states, next_decoder_cache

    def _prepare_decoder_attention_mask(self, attention_mask, input_shape, inputs_embeds, past_key_values_length):
        # Create causal mask
        # [bsz, seq_len] -> [bsz, 1, tgt_seq_len, src_seq_len]
        bsz, tgt_len = input_shape
        device = inputs_embeds.device
        
        mask = torch.full((tgt_len, tgt_len), torch.finfo(inputs_embeds.dtype).min, device=device)
        mask_cond = torch.arange(mask.size(-1), device=device)
        mask.masked_fill_(mask_cond < (mask_cond + 1).view(mask.size(-1), 1), 0)
        mask = mask.to(inputs_embeds.dtype)
        
        if past_key_values_length > tgt_len:
            mask = torch.cat([torch.zeros(tgt_len, past_key_values_length - tgt_len, dtype=mask.dtype, device=device), mask], dim=-1)
            
        mask = mask[None, None, :, :].expand(bsz, 1, tgt_len, past_key_values_length)

        if attention_mask is not None:
            # attention_mask is [bsz, src_seq_len]
            attention_mask = attention_mask.unsqueeze(1).unsqueeze(2) # [bsz, 1, 1, src_seq_len]
            # convert 0 to min, 1 to 0
            attention_mask = (1.0 - attention_mask) * torch.finfo(inputs_embeds.dtype).min
            mask = mask + attention_mask

        return mask

class VajraForCausalLM(nn.Module):
    def __init__(self, config: VajraConfig):
        super().__init__()
        self.config = config
        self.model = VajraModel(config)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        
        if config.tie_word_embeddings:
            self.lm_head.weight = self.model.embed_tokens.weight

    def forward(
        self,
        input_ids: torch.LongTensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[Tuple[torch.Tensor, torch.Tensor]]] = None,
        use_cache: Optional[bool] = None,
        labels: Optional[torch.LongTensor] = None,
    ) -> Union[Tuple[torch.Tensor, ...], dict]:
        
        hidden_states, past_key_values = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            use_cache=use_cache,
        )

        logits = self.lm_head(hidden_states)

        loss = None
        if labels is not None:
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(shift_logits.view(-1, self.config.vocab_size), shift_labels.view(-1))

        if loss is not None:
            return {"loss": loss, "logits": logits, "past_key_values": past_key_values}
        return {"logits": logits, "past_key_values": past_key_values}
