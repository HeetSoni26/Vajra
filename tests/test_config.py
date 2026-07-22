from model import ModelConfig


def test_head_dim():
    cfg = ModelConfig(hidden_size=128, num_attention_heads=4)
    assert cfg.head_dim == 32
