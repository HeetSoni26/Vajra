import tempfile

import pytest

from tokenizer.configs.settings import TokenizerConfig
from tokenizer.tokenizers.hf_bpe import HFBpeTokenizer
from tokenizer.trainers.hf_trainer import HFBpeTrainer

pytest.importorskip("tokenizers")


def test_hf_bpe_training():
    config = TokenizerConfig(vocab_size=100)
    trainer = HFBpeTrainer(config)

    docs = ["This is the first sentence for Vajra.", "Vajra is a great foundation model."]
    tokenizer = trainer.train(docs)

    assert tokenizer.get_vocab_size() <= 100

    with tempfile.TemporaryDirectory() as d:
        tokenizer.save_pretrained(d)

        # Reload
        loaded = HFBpeTokenizer.from_pretrained(d)
        assert loaded.get_vocab_size() == tokenizer.get_vocab_size()

        # Test encode
        encoded = loaded.encode("Vajra is great")
        assert len(encoded) > 0

        # Test decode
        decoded = loaded.decode(encoded)
        assert "Vajra" in decoded


def test_hf_benchmark():
    config = TokenizerConfig(vocab_size=100)
    trainer = HFBpeTrainer(config)
    docs = ["Benchmark sentence one", "Benchmark sentence two"]
    tokenizer = trainer.train(docs)

    from tokenizer.statistics.benchmark import TokenizerBenchmark

    bench = TokenizerBenchmark(tokenizer)

    res = bench.benchmark_encoding(docs)
    assert res["total_texts"] == 2
    assert res["duration_seconds"] >= 0
