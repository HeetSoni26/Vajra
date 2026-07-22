import torch
import torch.nn.functional as F
from typing import Optional
from model.modeling import VajraForCausalLM

class GenerationEngine:
    """
    Inference engine for Vajra models supporting greedy and sampling strategies.
    """
    def __init__(self, model: VajraForCausalLM):
        self.model = model
        self.model.eval()
        self.device = next(self.model.parameters()).device
        self.vocab_size = self.model.config.vocab_size

    @torch.no_grad()
    def generate(
        self,
        input_ids: torch.LongTensor,
        max_new_tokens: int = 20,
        temperature: float = 1.0,
        top_k: int = 0,
        top_p: float = 1.0,
        eos_token_id: Optional[int] = None,
        use_cache: bool = True
    ) -> torch.LongTensor:
        
        # input_ids: [bsz, seq_len]
        bsz = input_ids.shape[0]
        past_key_values = None
        
        unfinished_sequences = torch.ones(bsz, dtype=torch.bool, device=self.device)
        
        for _ in range(max_new_tokens):
            if past_key_values is not None and use_cache:
                # only pass the last token
                model_inputs = input_ids[:, -1:]
            else:
                model_inputs = input_ids

            outputs = self.model(
                input_ids=model_inputs,
                past_key_values=past_key_values,
                use_cache=use_cache
            )
            
            logits = outputs["logits"]
            past_key_values = outputs["past_key_values"]
            
            next_token_logits = logits[:, -1, :]
            
            if temperature != 1.0:
                next_token_logits = next_token_logits / temperature
                
            next_token = self._sample(next_token_logits, top_k, top_p)
            
            input_ids = torch.cat([input_ids, next_token[:, None]], dim=-1)
            
            if eos_token_id is not None:
                unfinished_sequences = unfinished_sequences & (next_token != eos_token_id)
                if not unfinished_sequences.any():
                    break
                    
        return input_ids

    def _sample(self, logits: torch.Tensor, top_k: int, top_p: float) -> torch.Tensor:
        if top_k > 0:
            indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
            logits[indices_to_remove] = float('-inf')
            
        if top_p < 1.0:
            sorted_logits, sorted_indices = torch.sort(logits, descending=True)
            cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
            sorted_indices_to_remove = cumulative_probs > top_p
            sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
            sorted_indices_to_remove[..., 0] = 0
            
            indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
            logits[indices_to_remove] = float('-inf')
            
        probs = F.softmax(logits, dim=-1)
        next_tokens = torch.multinomial(probs, num_samples=1).squeeze(1)
        return next_tokens
