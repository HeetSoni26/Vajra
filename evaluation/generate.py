import argparse
import json
import logging
from pathlib import Path
from typing import Any

from inference.engine import GenerationConfig, InferenceEngine
from utils.logging import setup_logger

logger = setup_logger("generate")


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate text using a Vajra checkpoint.")
    parser.add_argument("--checkpoint", required=True, help="Path to checkpoint .pt file or directory containing latest.pt")
    parser.add_argument("--config", required=True, help="Path to training config.yaml")
    parser.add_argument("--prompt", required=True, help="Input prompt")
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top-k", type=int, default=0)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--no-kv-cache", action="store_true", help="Disable KV cache")
    parser.add_argument("--stream", action="store_true", help="Enable streaming output")
    parser.add_argument("--output-dir", default=None, help="Directory to save generated sample (optional)")

    args = parser.parse_args()
    
    ckpt_path = Path(args.checkpoint)
    if ckpt_path.is_dir():
        ckpt_path = ckpt_path / "latest.pt"

    logger.info(f"Loading checkpoint from: {ckpt_path}")
    engine = InferenceEngine.from_config(args.config, checkpoint=str(ckpt_path))

    gen_cfg = GenerationConfig(
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_k=args.top_k,
        top_p=args.top_p,
        use_kv_cache=not args.no_kv_cache,
    )

    logger.info(f"Generating from prompt: {args.prompt}")
    
    generated_text = ""
    if args.stream:
        print(f"Prompt: {args.prompt}")
        print("Completion: ", end="", flush=True)
        for token in engine.generate_stream(args.prompt, gen_cfg):
            print(token, end="", flush=True)
            generated_text += token
        print("\n")
    else:
        results = engine.generate([args.prompt], gen_cfg)
        generated_text = results[0]
        print(f"\nPrompt: {args.prompt}")
        print(f"Completion:\n{generated_text}\n")

    if args.output_dir:
        out_dir = Path(args.output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        sample_path = out_dir / "samples.txt"
        with sample_path.open("a", encoding="utf-8") as f:
            f.write(f"--- Prompt ---\n{args.prompt}\n")
            f.write(f"--- Completion ---\n{generated_text}\n\n")
        logger.info(f"Sample saved to {sample_path}")

if __name__ == "__main__":
    main()
