import argparse

from model.config import get_preset
from model.modeling import VajraForCausalLM
from training.config import TrainingConfig
from training.engine.loop import TrainingEngine

def cmd_train(args):
    print(f"Loading Configuration from preset: {args.preset}")
    model_config = get_preset(args.preset)
    
    train_config = TrainingConfig(
        dataset_dir=args.dataset_dir,
        output_dir=args.output_dir,
        batch_size=args.batch_size,
        max_steps=args.max_steps,
        mixed_precision=args.mixed_precision
    )
    
    if args.config_file:
        train_config = TrainingConfig.load(args.config_file)
        
    model = VajraForCausalLM(model_config)
    engine = TrainingEngine(model, train_config)
    
    engine.train(resume_from_checkpoint=args.resume_from)

def cmd_dry_run(args):
    print("Performing Dry Run...")
    model_config = get_preset("Vajra-370M")
    train_config = TrainingConfig(batch_size=2, max_steps=5, dataset_dir=args.dataset_dir, output_dir=args.output_dir)
    
    model = VajraForCausalLM(model_config)
    TrainingEngine(model, train_config)
    
    print("Dry Run initialized successfully. Testing dataloader...")
    from training.data.loader import create_dataloader
    dl = create_dataloader(train_config.dataset_dir, 2, 2048)
    batch = next(iter(dl))
    print(f"Batch shape: {batch.shape}")
    print("Dry run passed.")

def main():
    parser = argparse.ArgumentParser(description="Vajra Single-GPU Training CLI")
    subparsers = parser.add_subparsers(dest="command", required=True)
    
    train_parser = subparsers.add_parser("train")
    train_parser.add_argument("--preset", default="Vajra-370M", help="Model architecture preset")
    train_parser.add_argument("--dataset-dir", required=True, help="Directory containing binary shards")
    train_parser.add_argument("--output-dir", required=True, help="Output directory for checkpoints")
    train_parser.add_argument("--config-file", help="Path to TrainingConfig JSON")
    train_parser.add_argument("--resume-from", help="Path to checkpoint directory to resume from")
    train_parser.add_argument("--batch-size", type=int, default=8)
    train_parser.add_argument("--max-steps", type=int, default=10000)
    train_parser.add_argument("--mixed-precision", choices=["none", "fp16", "bf16"], default="bf16")
    
    dry_parser = subparsers.add_parser("dry-run")
    dry_parser.add_argument("--dataset-dir", required=True)
    dry_parser.add_argument("--output-dir", default="output/dry_run")
    
    args = parser.parse_args()
    
    if args.command == "train":
        cmd_train(args)
    elif args.command == "dry-run":
        cmd_dry_run(args)

if __name__ == "__main__":
    main()
