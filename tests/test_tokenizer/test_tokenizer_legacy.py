from __future__ import annotations

from pathlib import Path

import pytest
from tokenizers import ByteLevelBPETokenizer
from transformers import AutoTokenizer, PreTrainedTokenizerFast


def test_tokenizer_training_and_serialization(tmp_path: Path):
    pytest.importorskip("tokenizers")
    pytest.importorskip("transformers")
    # Prepare dummy corpus file
    corpus_file = tmp_path / "corpus.jsonl"
    corpus_file.write_text('{"text": "Transformers use attention and RMSNorm blocks."}\n')

    out_dir = tmp_path / "tokenizer_out"
    out_dir.mkdir(parents=True, exist_ok=True)

    # Train tokenizer
    tokenizer = ByteLevelBPETokenizer(add_prefix_space=False)
    special_tokens = ["<|pad|>", "<|bos|>", "<|eos|>", "<|unk|>"]
    tokenizer.train(
        files=[str(corpus_file)],
        vocab_size=100,
        min_frequency=1,
        special_tokens=special_tokens,
    )
    tokenizer.save_model(str(out_dir))
    tokenizer.save(str(out_dir / "tokenizer.json"))

    assert (out_dir / "tokenizer.json").exists()
    assert (out_dir / "vocab.json").exists()
    assert (out_dir / "merges.txt").exists()

    # Load with PreTrainedTokenizerFast
    fast_tok = PreTrainedTokenizerFast(
        tokenizer_file=str(out_dir / "tokenizer.json"),
        bos_token="<|bos|>",
        eos_token="<|eos|>",
        unk_token="<|unk|>",
        pad_token="<|pad|>",
    )
    fast_tok.save_pretrained(out_dir)

    # AutoTokenizer loading verification
    loaded_tok = AutoTokenizer.from_pretrained(out_dir)
    sample_text = "Transformers attention"
    encoded = loaded_tok.encode(sample_text)
    decoded = loaded_tok.decode(encoded)

    assert len(encoded) > 0
    assert "Transformers" in decoded
