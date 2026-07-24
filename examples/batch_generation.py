"""Batch text generation example using FoundationLM."""

from inference.engine import GenerationConfig, InferenceEngine


def main():
    engine = InferenceEngine.from_config("configs/training/pretrain_tiny.yaml")

    prompts = [
        "Language models are",
        "Deep learning enables",
        "The universe consists of",
    ]

    gen_cfg = GenerationConfig(max_new_tokens=24, temperature=0.7)
    results = engine.generate(prompts, gen_cfg)

    for p, r in zip(prompts, results):
        print(f"Prompt: {p}")
        print(f"Result: {r}\n---")


if __name__ == "__main__":
    main()
