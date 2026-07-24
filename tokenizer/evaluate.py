from __future__ import annotations

import argparse

from transformers import AutoTokenizer


def chars_per_token(tokenizer, text: str) -> float:
    tokens = tokenizer.encode(text)
    return len(text) / max(1, len(tokens))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tokenizer_dir", default="tokenizer/v1.0")
    parser.add_argument(
        "--sample", default="The quick brown fox writes Python: def f(x): return x + 1"
    )
    args = parser.parse_args()
    tok = AutoTokenizer.from_pretrained(args.tokenizer_dir)
    encoded = tok.encode(args.sample)
    decoded = tok.decode(encoded)
    report = {
        "chars_per_token": chars_per_token(tok, args.sample),
        "roundtrip_ok": decoded == args.sample,
        "num_tokens": len(encoded),
    }
    print(report)


if __name__ == "__main__":
    main()
