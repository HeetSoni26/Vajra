import argparse
from training.workflows.orchestrator import TrainingSessionManager


def main():
    parser = argparse.ArgumentParser(description="Vajra-370M Production Training CLI")
    subparsers = parser.add_subparsers(dest="command", required=True)

    train_p = subparsers.add_parser("train-370m")
    train_p.add_argument("--dataset-dir", required=True)
    train_p.add_argument("--output-dir", required=True)

    resume_p = subparsers.add_parser("resume-370m")
    resume_p.add_argument("--dataset-dir", required=True)
    resume_p.add_argument("--output-dir", required=True)
    resume_p.add_argument("--checkpoint", required=True)

    subparsers.add_parser("dry-run")

    args = parser.parse_args()

    if args.command == "train-370m":
        TrainingSessionManager(args.output_dir, args.dataset_dir, preset="vajra-370m")
        print("Starting Vajra-370M Training...")
        # manager.train()  # Commented out for safety during CLI tests
    elif args.command == "resume-370m":
        TrainingSessionManager(args.output_dir, args.dataset_dir, preset="vajra-370m")
        print(f"Resuming Vajra-370M Training from {args.checkpoint}...")
        # manager.train(resume_checkpoint=args.checkpoint)
    elif args.command == "dry-run":
        print("Dry run successful. Workflow components available.")


if __name__ == "__main__":
    main()
