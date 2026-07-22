"""Basic text generation example using FoundationLM InferenceEngine."""

from inference.engine import InferenceEngine, GenerationConfig


def main():
    # Load inference engine from pretraining config
    engine = InferenceEngine.from_config("configs/training/pretrain_tiny.yaml")

    prompt = "Artificial Intelligence is transforming"
    gen_cfg = GenerationConfig(
        max_new_tokens=32,
        temperature=0.7,
        top_k=50,
        top_p=0.9,
        use_kv_cache=True,
    )

    print(f"Prompt: {prompt}\n---")
    results = engine.generate(prompt, gen_cfg)
    print(results[0])


if __name__ == "__main__":
    main()
