import argparse
from training.production.config import ProductionConfig

def cmd_dry_run(args):
    print("Initializing production configuration...")
    ProductionConfig(dataset_dir="dummy", output_dir="dummy")
    print("Configuration loaded. All modules discovered.")
    
def main():
    parser = argparse.ArgumentParser(description="Vajra Production Training CLI")
    subparsers = parser.add_subparsers(dest="command", required=True)
    
    train_p = subparsers.add_parser("train-production")
    train_p.add_argument("--dataset-dir", required=True)
    train_p.add_argument("--output-dir", required=True)
    
    subparsers.add_parser("dry-run")
    
    args = parser.parse_args()
    
    if args.command == "train-production":
        print("Production training launch not yet implemented for full dataset.")
    elif args.command == "dry-run":
        cmd_dry_run(args)

if __name__ == "__main__":
    main()
