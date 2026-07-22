import torch


@torch.no_grad()
def generate(model, input_ids: torch.Tensor, max_new_tokens: int = 128, temperature: float = 0.8):
    model.eval()
    for _ in range(max_new_tokens):
        logits = model(input_ids)["logits"][:, -1, :]
        if temperature <= 0:
            next_token = logits.argmax(dim=-1, keepdim=True)
        else:
            probs = torch.softmax(logits / temperature, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
        input_ids = torch.cat([input_ids, next_token], dim=-1)
    return input_ids
