import pytest
from model import FoundationLM, ModelConfig


def test_forward_shape():
    torch = pytest.importorskip("torch")

    cfg = ModelConfig(
        vocab_size=128,
        hidden_size=64,
        num_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        intermediate_size=128,
        max_position_embeddings=32,
        use_flash_attention=False,
    )
    model = FoundationLM(cfg)
    input_ids = torch.randint(0, cfg.vocab_size, (2, 16))
    out = model(input_ids, labels=input_ids)
    assert out["logits"].shape == (2, 16, cfg.vocab_size)
    assert out["loss"].ndim == 0
