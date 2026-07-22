from __future__ import annotations

import argparse


def format_prompt(message: str, system: str = "You are a helpful assistant.") -> str:
    return f"<|sys|>{system}<|user|>{message}<|assistant|>"


def main() -> None:
    parser = argparse.ArgumentParser(description="Minimal interactive chat CLI.")
    parser.add_argument("--model", default="checkpoints/final/hf")
    args = parser.parse_args()
    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError as exc:
        raise SystemExit("Install inference dependencies: pip install transformers torch") from exc
    tok = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype="auto", device_map="auto")
    while True:
        message = input("user> ").strip()
        if message in {"exit", "quit"}:
            break
        inputs = tok(format_prompt(message), return_tensors="pt").to(model.device)
        out = model.generate(**inputs, max_new_tokens=256, do_sample=True, temperature=0.7)
        print(tok.decode(out[0], skip_special_tokens=False))


if __name__ == "__main__":
    main()
