"""Interactive chat example using FoundationLM."""

from inference.engine import GenerationConfig, InferenceEngine


def main():
    engine = InferenceEngine.from_config("configs/training/pretrain_tiny.yaml")
    gen_cfg = GenerationConfig(max_new_tokens=64, temperature=0.7, top_k=50)

    messages = [
        "User: Hello! What can you do?\nAssistant:",
    ]

    for prompt in messages:
        print(prompt, end="")
        output = engine.generate(prompt, gen_cfg)[0]
        print(output)


if __name__ == "__main__":
    main()
