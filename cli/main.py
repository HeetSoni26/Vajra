"""Foundation LM CLI — Unified command-line interface."""

from __future__ import annotations

import argparse
import json
import sys

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")


def cmd_generate(args: argparse.Namespace) -> None:
    """Generate text from a prompt."""
    from inference.engine import InferenceEngine, GenerationConfig

    engine = InferenceEngine.from_config(args.config, args.checkpoint or None)
    gen_cfg = GenerationConfig(
        max_new_tokens=args.max_tokens,
        temperature=args.temperature,
        top_k=args.top_k,
        top_p=args.top_p,
        repetition_penalty=args.repetition_penalty,
        do_sample=(args.temperature > 0),
        seed=args.seed,
        use_kv_cache=not args.no_kv_cache,
    )

    if args.stream:
        print(f"Prompt: {args.prompt}\n---")
        for token in engine.generate_stream(args.prompt, gen_cfg):
            sys.stdout.write(token)
            sys.stdout.flush()
        print()
    else:
        results = engine.generate(args.prompt, gen_cfg)
        print(results[0])


def cmd_chat(args: argparse.Namespace) -> None:
    """Interactive chat REPL."""
    from inference.engine import InferenceEngine, GenerationConfig

    engine = InferenceEngine.from_config(args.config, args.checkpoint or None)
    gen_cfg = GenerationConfig(
        max_new_tokens=args.max_tokens,
        temperature=args.temperature,
        top_k=args.top_k,
        top_p=args.top_p,
        do_sample=True,
        use_kv_cache=True,
    )

    print("Foundation LM Chat (type 'exit' or 'quit' to leave)")
    while True:
        try:
            user_input = input("user> ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if user_input.lower() in {"exit", "quit", ""}:
            break

        prompt = f"<|user|>{user_input}<|assistant|>"
        results = engine.generate(prompt, gen_cfg)
        print(f"assistant> {results[0]}")


def cmd_evaluate(args: argparse.Namespace) -> None:
    """Run model evaluation on memmap datasets."""
    from evaluation.evaluator import ModelEvaluator
    from utils.file_utils import write_json

    evaluator = ModelEvaluator.from_config(args.config, args.checkpoint)
    report = evaluator.evaluate_all(
        data_dir=args.data_dir,
        sequence_length=args.sequence_length,
        batch_size=args.batch_size,
        max_batches=args.max_batches,
    )

    write_json(report, args.output)
    print(json.dumps(report, indent=2))


def cmd_tokenize(args: argparse.Namespace) -> None:
    """Tokenize or detokenize text."""
    from tokenizers import Tokenizer
    from pathlib import Path

    tok_path = Path(args.tokenizer)
    if tok_path.is_dir():
        tok_path = tok_path / "tokenizer.json"
    tokenizer = Tokenizer.from_file(str(tok_path))

    if args.decode:
        ids = json.loads(args.text)
        print(tokenizer.decode(ids))
    else:
        enc = tokenizer.encode(args.text)
        print(json.dumps({"ids": enc.ids, "tokens": enc.tokens, "num_tokens": len(enc.ids)}, indent=2))


def cmd_profile(args: argparse.Namespace) -> None:
    """Run performance benchmarks."""
    from training.profiler import run_benchmark

    report = run_benchmark(
        config_path=args.config,
        precision=args.precision,
        use_gradient_checkpointing=args.gradient_checkpointing,
        output_report=args.output,
    )
    print(json.dumps(report, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="vajra-lm",
        description="Foundation LM — CLI for generation, evaluation, and profiling",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    # --- generate ---
    p_gen = subparsers.add_parser("generate", help="Generate text from a prompt")
    p_gen.add_argument("--prompt", required=True)
    p_gen.add_argument("--config", default="configs/training/pretrain_tiny.yaml")
    p_gen.add_argument("--checkpoint", default=None)
    p_gen.add_argument("--max_tokens", type=int, default=64)
    p_gen.add_argument("--temperature", type=float, default=0.8)
    p_gen.add_argument("--top_k", type=int, default=50)
    p_gen.add_argument("--top_p", type=float, default=0.9)
    p_gen.add_argument("--repetition_penalty", type=float, default=1.0)
    p_gen.add_argument("--seed", type=int, default=None)
    p_gen.add_argument("--stream", action="store_true")
    p_gen.add_argument("--no_kv_cache", action="store_true")
    p_gen.set_defaults(func=cmd_generate)

    # --- chat ---
    p_chat = subparsers.add_parser("chat", help="Interactive chat REPL")
    p_chat.add_argument("--config", default="configs/training/pretrain_tiny.yaml")
    p_chat.add_argument("--checkpoint", default=None)
    p_chat.add_argument("--max_tokens", type=int, default=128)
    p_chat.add_argument("--temperature", type=float, default=0.7)
    p_chat.add_argument("--top_k", type=int, default=50)
    p_chat.add_argument("--top_p", type=float, default=0.9)
    p_chat.set_defaults(func=cmd_chat)

    # --- evaluate ---
    p_eval = subparsers.add_parser("evaluate", help="Evaluate model on datasets")
    p_eval.add_argument("--config", default="configs/training/pretrain_tiny.yaml")
    p_eval.add_argument("--checkpoint", default=None)
    p_eval.add_argument("--data_dir", default="data/tokenized")
    p_eval.add_argument("--sequence_length", type=int, default=128)
    p_eval.add_argument("--batch_size", type=int, default=4)
    p_eval.add_argument("--max_batches", type=int, default=None)
    p_eval.add_argument("--output", default="evaluation_report.json")
    p_eval.set_defaults(func=cmd_evaluate)

    # --- tokenize ---
    p_tok = subparsers.add_parser("tokenize", help="Tokenize or detokenize text")
    p_tok.add_argument("--text", required=True)
    p_tok.add_argument("--tokenizer", default="tokenizer")
    p_tok.add_argument("--decode", action="store_true", help="Decode token IDs to text")
    p_tok.set_defaults(func=cmd_tokenize)

    # --- profile ---
    p_prof = subparsers.add_parser("profile", help="Run performance benchmarks")
    p_prof.add_argument("--config", default="configs/training/pretrain_tiny.yaml")
    p_prof.add_argument("--precision", default="fp32", choices=["fp32", "fp16", "bf16"])
    p_prof.add_argument("--gradient_checkpointing", action="store_true")
    p_prof.add_argument("--output", default="benchmark_report.json")
    p_prof.set_defaults(func=cmd_profile)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
