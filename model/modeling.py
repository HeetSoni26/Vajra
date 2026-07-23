import torch
import torch.nn as nn
import torch.utils.checkpoint
from pathlib import Path
from typing import Optional, Tuple, List, Union, Any
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
        kv_cache: Optional[Any] = None,
        start_pos: int = 0,
        **kwargs,
    ) -> Tuple[torch.Tensor, Optional[List[Tuple[torch.Tensor, torch.Tensor]]]]:
        bsz, seq_len = input_ids.shape

        if kv_cache is not None and past_key_values is None:
            use_cache = True
            past_key_values = []
            for i in range(self.config.num_layers):
                k, v = kv_cache.get(i)
                if k is not None and v is not None:
                    # KVCache stores [bsz, num_kv_heads, seq_len, head_dim]
                    # VajraAttention uses [bsz, seq_len, num_kv_heads, head_dim]
                    past_key_values.append((k.transpose(1, 2), v.transpose(1, 2)))
                else:
                    past_key_values.append(None)

        if position_ids is None:
            device = input_ids.device
            past_key_values_length = start_pos
            if past_key_values is not None and past_key_values[0] is not None:
                past_key_values_length = past_key_values[0][0].shape[1]
            position_ids = torch.arange(
                past_key_values_length, seq_len + past_key_values_length, dtype=torch.long, device=device
            )
            position_ids = position_ids.unsqueeze(0).view(-1, seq_len)

        hidden_states = self.embed_tokens(input_ids)
        
        # Prepare RoPE
        kv_seq_len = hidden_states.shape[1]
        if past_key_values is not None and past_key_values[0] is not None:
            kv_seq_len += past_key_values[0][0].shape[1]
        elif start_pos > 0:
            kv_seq_len += start_pos
            
        cos, sin = self.rotary_emb(hidden_states, seq_len=kv_seq_len)
        
        # Prepare causal attention mask
        if seq_len > 1 or attention_mask is not None:
            attn_mask = self._prepare_decoder_attention_mask(
                attention_mask, (bsz, seq_len), hidden_states, kv_seq_len
            )
        else:
            attn_mask = None

        next_decoder_cache = () if use_cache else None

        use_checkpointing = getattr(self.config, "use_gradient_checkpointing", False) and self.training

        for idx, decoder_layer in enumerate(self.layers):
            past_key_value = past_key_values[idx] if past_key_values is not None else None

            if use_checkpointing:
                def create_custom_forward(module):
                    def custom_forward(*inputs):
                        return module(*inputs)
                    return custom_forward

                layer_outputs = torch.utils.checkpoint.checkpoint(
                    create_custom_forward(decoder_layer),
                    hidden_states,
                    attn_mask,
                    position_ids,
                    past_key_value,
                    False,
                    use_cache,
                    cos,
                    sin,
                    use_reentrant=False,
                )
            else:
                layer_outputs = decoder_layer(
                    hidden_states,
                    attention_mask=attn_mask,
                    position_ids=position_ids,
                    past_key_value=past_key_value,
                    output_attentions=False,
                    use_cache=use_cache,
                    cos=cos,
                    sin=sin,
                )

            hidden_states = layer_outputs[0]

            if use_cache:
                present_kv = layer_outputs[-1]
                next_decoder_cache += (present_kv,)
                if kv_cache is not None and present_kv is not None:
                    k_full, v_full = present_kv
                    # k_full: [bsz, full_seq_len, num_kv_heads, head_dim]
                    # Update KVCache directly with the full key and value states
                    kv_cache._keys[idx] = k_full.transpose(1, 2)
                    kv_cache._values[idx] = v_full.transpose(1, 2)
                    if idx == 0:
                        kv_cache._seq_len = k_full.shape[1]

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
        kv_cache: Optional[Any] = None,
        start_pos: int = 0,
        **kwargs,
    ) -> Union[Tuple[torch.Tensor, ...], dict]:
        
        hidden_states, past_key_values = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            use_cache=use_cache,
            kv_cache=kv_cache,
            start_pos=start_pos,
        )

        logits = self.lm_head(hidden_states)

        loss = None
        if labels is not None:
            assert labels.shape == input_ids.shape, (
                f"Labels shape {labels.shape} must match input_ids shape {input_ids.shape}. "
                "Note: Internal label shifting is handled automatically inside VajraForCausalLM.forward."
            )
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(shift_logits.view(-1, self.config.vocab_size), shift_labels.view(-1))

        if loss is not None:
            return {"loss": loss, "logits": logits, "past_key_values": past_key_values}
        return {"logits": logits, "past_key_values": past_key_values}

    def save_pretrained(self, save_directory: Union[str, Path], tokenizer_dir: Optional[Union[str, Path]] = None, **kwargs) -> dict:
        from inference.hf_compat import save_pretrained as _save_pretrained
        return _save_pretrained(self, save_directory, tokenizer_dir=tokenizer_dir)

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path: Union[str, Path], device: Union[torch.device, str] = "cpu", strict: bool = True, **kwargs) -> "VajraForCausalLM":
        from inference.hf_compat import load_pretrained as _load_pretrained
        model, _ = _load_pretrained(pretrained_model_name_or_path, device=device, strict=strict)
        return model

