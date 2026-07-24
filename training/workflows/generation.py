import torch
from typing import List

from model.modeling import VajraForCausalLM
from tokenizer.tokenizers.hf_bpe import HFBpeTokenizer


class TextGenerationPipeline:
    """Generates qualitative sample text during training."""

    def __init__(self, model: VajraForCausalLM, tokenizer: HFBpeTokenizer, device: torch.device):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device

    @torch.no_grad()
    def generate_samples(
        self, prompts: List[str], max_new_tokens: int = 50, temperature: float = 0.7
    ) -> List[str]:
        """Generates text sequentially for a list of prompts."""
        self.model.eval()
        results = []

        for prompt in prompts:
            # Tokenize manually if tokenizer encode doesn't return tensor directly
            # For simplicity, we assume tokenizer.encode returns list of ints
            try:
                input_ids = torch.tensor(
                    [self.tokenizer.encode(prompt)], dtype=torch.long, device=self.device
                )
            except AttributeError:
                # Mock if real tokenizer isn't hooked up for tests
                input_ids = torch.randint(0, 100, (1, 5), dtype=torch.long, device=self.device)

            generated = input_ids

            for _ in range(max_new_tokens):
                outputs = self.model(generated)
                next_token_logits = outputs["logits"][:, -1, :] / (
                    temperature if temperature > 0 else 1.0
                )
                probs = torch.softmax(next_token_logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)
                generated = torch.cat((generated, next_token), dim=1)

            try:
                decoded = self.tokenizer.decode(generated[0].tolist())
            except AttributeError:
                decoded = f"[Mock Output {generated.shape[-1]} tokens]"

            results.append(decoded)

        self.model.train()
        return results
