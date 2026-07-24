import warnings

import torch
import torch.nn.functional as F


def _scaled_dot_product_attention_fallback(q, k, v, attn_mask=None, dropout_p=0.0, is_causal=False):
    """Fallback if flash attention is not available natively or via SDP."""
    B, num_heads, L, head_dim = q.shape
    scale = 1.0 / (head_dim**0.5)

    scores = torch.matmul(q, k.transpose(-2, -1)) * scale
    if is_causal:
        causal_mask = torch.tril(torch.ones(L, L, device=q.device)).view(1, 1, L, L)
        scores = scores.masked_fill(causal_mask == 0, float("-inf"))
    if attn_mask is not None:
        scores = scores + attn_mask

    attn_weights = F.softmax(scores, dim=-1)
    if dropout_p > 0.0:
        attn_weights = F.dropout(attn_weights, p=dropout_p)

    return torch.matmul(attn_weights, v)


def apply_flash_attention(
    q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, is_causal: bool = True
) -> torch.Tensor:
    """
    Applies FlashAttention if available via PyTorch's scaled_dot_product_attention.
    Otherwise falls back to manual implementation.
    """
    if hasattr(F, "scaled_dot_product_attention"):
        # PyTorch 2.0+ native SDP automatically uses FlashAttention if applicable.
        try:
            return F.scaled_dot_product_attention(
                q, k, v, attn_mask=None, dropout_p=0.0, is_causal=is_causal
            )
        except Exception as e:
            warnings.warn(f"Flash Attention via SDP failed ({e}). Falling back.")
            return _scaled_dot_product_attention_fallback(q, k, v, is_causal=is_causal)
    else:
        return _scaled_dot_product_attention_fallback(q, k, v, is_causal=is_causal)
