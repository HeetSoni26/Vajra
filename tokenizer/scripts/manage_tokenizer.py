import argparse
from tokenizer.configs.settings import TokenizerConfig
from tokenizer.tokenizers.hf_bpe import HFBpeTokenizer
from tokenizer.trainers.hf_trainer import HFBpeTrainer
from tokenizer.statistics.benchmark import TokenizerBenchmark
from tokenizer.validators.validator import TokenizerValidator
from tokenizer.vocab.manager import VocabularyManager


def cmd_train(args):
    print("Initializing HFBpeTrainer...")
    config = TokenizerConfig()
    trainer = HFBpeTrainer(config)

    if args.dry_run:
        print("Dry-run: Would train tokenizer here.")
        return

    print("Training mock tokenizer on a sample sentence...")
    sample = [
        "Hello world, this is Vajra.",
        "Vajra is a foundation model.",
        "The tokenizer is training.",
    ]
    tokenizer = trainer.train(sample)

    out_dir = config.output_dir
    tokenizer.save_pretrained(out_dir)
    print(f"Tokenizer trained and saved to {out_dir}")


def cmd_validate(args):
    print("Validating tokenizer...")
    config = TokenizerConfig()
    try:
        tokenizer = HFBpeTokenizer.from_pretrained(config.output_dir)
    except Exception as e:
        print(f"Failed to load tokenizer: {e}")
        return

    # We fake the vocab manager since HF has internal vocab, but we can pass mock for interface
    manager = VocabularyManager(config)
    manager.token_to_id = tokenizer._tokenizer.get_vocab()

    validator = TokenizerValidator(tokenizer, manager)
    valid = validator.validate_round_trip("Hello Vajra")
    print(f"Round-trip validation: {'PASSED' if valid else 'FAILED'}")


def cmd_encode(args):
    config = TokenizerConfig()
    tokenizer = HFBpeTokenizer.from_pretrained(config.output_dir)
    encoded = tokenizer.encode(args.text)
    print(f"Encoded: {encoded}")


def cmd_decode(args):
    config = TokenizerConfig()
    tokenizer = HFBpeTokenizer.from_pretrained(config.output_dir)
    decoded = tokenizer.decode(args.ids)
    print(f"Decoded: {decoded}")


def cmd_stats(args):
    config = TokenizerConfig()
    tokenizer = HFBpeTokenizer.from_pretrained(config.output_dir)
    print(f"Vocabulary Size: {tokenizer.get_vocab_size()}")


def cmd_benchmark(args):
    config = TokenizerConfig()
    tokenizer = HFBpeTokenizer.from_pretrained(config.output_dir)
    benchmark = TokenizerBenchmark(tokenizer)

    texts = [
        "This is a test of the vajra encoding speed.",
        "It should be extremely fast because it uses rust.",
    ] * 1000
    res = benchmark.benchmark_encoding(texts)

    import json

    print(json.dumps(res, indent=2))


def main():
    parser = argparse.ArgumentParser(description="Vajra Tokenizer CLI")
    subparsers = parser.add_subparsers(dest="command", required=True)

    # Train command
    train_parser = subparsers.add_parser("train", help="Train a new tokenizer")
    train_parser.add_argument("--dry-run", action="store_true")

    # Validate command
    validate_parser = subparsers.add_parser("validate", help="Validate tokenizer integrity")
    validate_parser.add_argument("--dry-run", action="store_true")

    # Encode
    encode_parser = subparsers.add_parser("encode", help="Encode text")
    encode_parser.add_argument("text", type=str)

    # Decode
    decode_parser = subparsers.add_parser("decode", help="Decode IDs")
    decode_parser.add_argument("ids", type=int, nargs="+")

    # Stats
    subparsers.add_parser("stats", help="Show vocabulary and token statistics")

    args = parser.parse_args()

    if args.command == "train":
        cmd_train(args)
    elif args.command == "validate":
        cmd_validate(args)
    elif args.command == "encode":
        cmd_encode(args)
    elif args.command == "decode":
        cmd_decode(args)
    elif args.command == "stats":
        cmd_stats(args)


if __name__ == "__main__":
    main()
