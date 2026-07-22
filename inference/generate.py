from __future__ import annotations

import argparse


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate text with a HuggingFace checkpoint.")
    parser.add_argument("--model", default="checkpoints/final/hf")
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--max_new_tokens", type=int, default=128)
    args = parser.parse_args()
    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError as exc:
        raise SystemExit("Install inference dependencies: pip install transformers torch") from exc
    tok = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype="auto", device_map="auto")
    inputs = tok(args.prompt, return_tensors="pt").to(model.device)
    out = model.generate(**inputs, max_new_tokens=args.max_new_tokens, do_sample=True, temperature=0.7)
    print(tok.decode(out[0], skip_special_tokens=False))


if __name__ == "__main__":
    main()
