import argparse
import sys
import json
from pathlib import Path

from evaluation.config import EvaluationConfig
from evaluation.engine.evaluator import Evaluator
from evaluation.validation.checker import ValidationEngine
from evaluation.reporting.comparison import ComparisonEngine


def cmd_evaluate(args):
    config = EvaluationConfig()
    if args.config_file:
        config = EvaluationConfig.load(args.config_file)

    evaluator = Evaluator(config)
    report_path = evaluator.evaluate(args.checkpoint_dir, args.dataset_dir)
    print(f"Evaluation report generated at {report_path}")


def cmd_compare(args):
    path_a = Path(args.report_a)
    path_b = Path(args.report_b)

    with open(path_a, "r") as f:
        rep_a = json.load(f)["metrics"]
    with open(path_b, "r") as f:
        rep_b = json.load(f)["metrics"]

    diff = ComparisonEngine.compare(rep_a, rep_b)
    print(json.dumps(diff, indent=2))


def cmd_validate(args):
    try:
        ValidationEngine.validate_checkpoint(args.checkpoint_dir)
        print("Checkpoint is structurally valid.")
    except Exception as e:
        print(f"Validation failed: {e}")
        sys.exit(1)


def main():
    parser = argparse.ArgumentParser(description="Vajra Evaluation CLI")
    subparsers = parser.add_subparsers(dest="command", required=True)

    eval_parser = subparsers.add_parser("evaluate")
    eval_parser.add_argument("--checkpoint-dir", required=True)
    eval_parser.add_argument("--dataset-dir", required=False)
    eval_parser.add_argument("--config-file", required=False)

    comp_parser = subparsers.add_parser("compare")
    comp_parser.add_argument("report_a", help="Path to JSON report A (baseline)")
    comp_parser.add_argument("report_b", help="Path to JSON report B (target)")

    val_parser = subparsers.add_parser("validate")
    val_parser.add_argument("checkpoint_dir")

    args = parser.parse_args()

    if args.command == "evaluate":
        cmd_evaluate(args)
    elif args.command == "compare":
        cmd_compare(args)
    elif args.command == "validate":
        cmd_validate(args)


if __name__ == "__main__":
    main()
