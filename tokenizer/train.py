from __future__ import annotations

import argparse
import glob
from pathlib import Path

import yaml
from tokenizers import ByteLevelBPETokenizer
from transformers import PreTrainedTokenizerFast


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/tokenizer.yaml")
    args = parser.parse_args()
    cfg = yaml.safe_load(Path(args.config).read_text())
    files = sorted(glob.glob(cfg["input_glob"]))
    if not files:
        raise FileNotFoundError(f"No tokenizer corpus files matched {cfg['input_glob']}")

    out = Path(cfg["output_dir"])
    out.mkdir(parents=True, exist_ok=True)
    tokenizer = ByteLevelBPETokenizer(add_prefix_space=False)
    tokenizer.train(
        files=files,
        vocab_size=int(cfg["vocab_size"]),
        min_frequency=int(cfg["min_frequency"]),
        special_tokens=cfg["special_tokens"],
    )
    tokenizer.save_model(str(out))
    tokenizer.save(str(out / "tokenizer.json"))

    fast = PreTrainedTokenizerFast(
        tokenizer_file=str(out / "tokenizer.json"),
        bos_token="<|bos|>",
        eos_token="<|eos|>",
        unk_token="<|unk|>",
        pad_token="<|pad|>",
    )
    fast.save_pretrained(out)


if __name__ == "__main__":
    main()
