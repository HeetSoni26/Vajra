from tokenizer.configs.settings import TokenizerConfig
from tokenizer.tokenizers.mock import MockTokenizer
from tokenizer.encoders.pipeline import TokenizationPipeline
from tokenizer.vocab.manager import VocabularyManager
from tokenizer.validators.validator import TokenizerValidator
from tokenizer.statistics.models import TokenizerStatistics
import tempfile
from pathlib import Path


def test_tokenizer_config():
    config = TokenizerConfig(vocab_size=10000)
    assert config.vocab_size == 10000
    assert config.bos_token == "<s>"


def test_mock_tokenizer_encoding():
    config = TokenizerConfig()
    tokenizer = MockTokenizer(config)

    text = "hello world"
    encoded = tokenizer.encode(text)
    assert len(encoded) == 2
    assert tokenizer.get_vocab_size() == 2

    decoded = tokenizer.decode(encoded)
    assert decoded == text


def test_tokenization_pipeline():
    config = TokenizerConfig()
    tokenizer = MockTokenizer(config)
    pipeline = TokenizationPipeline(tokenizer)

    stream = ["test one", "test two"]
    results = list(pipeline.encode_stream(stream))
    assert len(results) == 2

    batch_results = pipeline.encode_batch(stream)
    assert len(batch_results) == 2


def test_vocabulary_manager():
    config = TokenizerConfig()
    manager = VocabularyManager(config)

    id1 = manager.add_token("hello")
    id2 = manager.add_token("world")
    assert id1 != id2

    id1_dup = manager.add_token("hello")
    assert id1 == id1_dup

    with tempfile.TemporaryDirectory() as d:
        path = Path(d) / "vocab.json"
        manager.save(path)

        manager2 = VocabularyManager(config)
        manager2.load(path)
        assert manager2.token_to_id == manager.token_to_id


def test_validator():
    config = TokenizerConfig()
    tokenizer = MockTokenizer(config)

    manager = VocabularyManager(config)
    validator = TokenizerValidator(tokenizer, manager)

    # We haven't loaded special tokens in mock vocab manually, so this is false
    assert validator.validate_special_tokens(["<s>"]) is False

    manager.add_token("<s>")
    assert validator.validate_special_tokens(["<s>"]) is True

    # Round trip test
    text = "round trip test"
    assert validator.validate_round_trip(text) is True


def test_statistics():
    stats = TokenizerStatistics(total_characters=100, total_tokens=25, unknown_tokens_count=2)

    assert stats.compression_ratio == 4.0
    assert stats.unknown_token_frequency == 0.08
