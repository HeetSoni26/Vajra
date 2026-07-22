from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from transformers import AutoTokenizer


def compute_tokenizer_metrics(tokenizer: Any, text_samples: list[str], name: str = "tokenizer") -> dict[str, Any]:
    """Compute detailed evaluation metrics for a given tokenizer."""
    total_chars = 0
    total_words = 0
    total_tokens = 0
    unk_count = 0
    roundtrip_successes = 0

    unk_id = getattr(tokenizer, "unk_token_id", None)

    for text in text_samples:
        if not text.strip():
            continue
        total_chars += len(text)
        total_words += len(text.split())

        encoded = tokenizer.encode(text)
        total_tokens += len(encoded)

        if unk_id is not None:
            unk_count += encoded.count(unk_id)

        decoded = tokenizer.decode(encoded, skip_special_tokens=True)
        # Check roundtrip (normalized whitespace comparison if needed)
        if decoded.strip() == text.strip() or text in decoded:
            roundtrip_successes += 1

    num_samples = max(1, len(text_samples))
    vocab_size = getattr(tokenizer, "vocab_size", len(tokenizer.get_vocab()) if hasattr(tokenizer, "get_vocab") else 0)

    return {
        "name": name,
        "vocab_size": vocab_size,
        "sample_count": num_samples,
        "total_chars": total_chars,
        "total_words": total_words,
        "total_tokens": total_tokens,
        "compression_ratio": round(total_chars / max(1, total_tokens), 3),
        "fertility": round(total_tokens / max(1, total_words), 3),
        "unk_rate": round(unk_count / max(1, total_tokens), 5),
        "roundtrip_accuracy": round(roundtrip_successes / num_samples, 3),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze and benchmark trained tokenizer.")
    parser.add_argument("--tokenizer_dir", default="tokenizer/v1.0")
    parser.add_argument("--sample_file", default="data/tokenizer_corpus/sample.jsonl")
    parser.add_argument("--baseline", default="gpt2", help="Baseline HuggingFace model/tokenizer to compare against")
    parser.add_argument("--output_report", default="tokenizer/v1.0/validation_report.json")
    args = parser.parse_args()

    # Load corpus samples
    sample_path = Path(args.sample_file)
    texts: list[str] = []
    if sample_path.exists():
        for line in sample_path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                try:
                    data = json.loads(line)
                    if "text" in data:
                        texts.append(data["text"])
                except Exception:
                    texts.append(line)
    if not texts:
        texts = [
            "Transformers use self-attention mechanisms and multi-head GQA blocks.",
            "def fibonacci(n: int) -> int:\n    return n if n < 2 else fibonacci(n - 1) + fibonacci(n - 2)",
            "Mathematics expression: f(x) = \\int_{-\\infty}^{\\infty} e^{-x^2} dx = \\sqrt{\\pi}",
        ]

    # Evaluate target tokenizer
    print(f"Loading trained tokenizer from: {args.tokenizer_dir}")
    target_tok = AutoTokenizer.from_pretrained(args.tokenizer_dir)
    target_metrics = compute_tokenizer_metrics(target_tok, texts, name="FoundationLM Tokenizer")

    # Evaluate baseline tokenizer (e.g. gpt2)
    baseline_metrics: dict[str, Any] | None = None
    try:
        baseline_tok = AutoTokenizer.from_pretrained(args.baseline)
        baseline_metrics = compute_tokenizer_metrics(baseline_tok, texts, name=f"Baseline ({args.baseline})")
    except Exception as e:
        print(f"Warning: Could not load baseline tokenizer '{args.baseline}': {e}")

    # Sample tokenization comparison
    sample_text = texts[0]
    sample_encoded = target_tok.encode(sample_text)
    sample_decoded = target_tok.decode(sample_encoded)
    sample_tokens = target_tok.convert_ids_to_tokens(sample_encoded)

    report = {
        "target": target_metrics,
        "baseline": baseline_metrics,
        "qualitative_sample": {
            "input_text": sample_text,
            "token_ids": sample_encoded,
            "tokens": sample_tokens[:20],
            "decoded_output": sample_decoded,
        },
    }

    report_path = Path(args.output_report)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print("\n================ TOKENIZER VALIDATION REPORT ================")
    print(json.dumps(report, indent=2))
    print(f"Report saved to: {report_path}")


if __name__ == "__main__":
    main()
