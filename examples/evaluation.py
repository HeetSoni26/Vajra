"""Model evaluation example on tokenized memmap data."""

from evaluation.evaluator import ModelEvaluator


def main():
    evaluator = ModelEvaluator.from_config("configs/training/pretrain_tiny.yaml")
    report = evaluator.evaluate_all("data/tokenized", max_batches=5)

    print("Evaluation Results:")
    for split, metrics in report.get("splits", {}).items():
        print(f"  [{split}] Loss: {metrics['avg_cross_entropy']} | PPL: {metrics['perplexity']}")


if __name__ == "__main__":
    main()
