import argparse
import sys
import json
from pathlib import Path
from model.config import get_preset
from model.modeling import VajraForCausalLM
from model.utils import summarize_model, initialize_weights
from model.checkpoints import CheckpointManager

def cmd_create(args):
    print(f"Creating new model from preset: {args.preset}")
    try:
        config = get_preset(args.preset)
        model = VajraForCausalLM(config)
        initialize_weights(model, config)
        
        output_dir = Path(args.output_dir)
        CheckpointManager.save_checkpoint(model, output_dir, use_safetensors=args.use_safetensors)
        print(f"Model saved to {output_dir}")
    except ValueError as e:
        print(e)
        sys.exit(1)

def cmd_summary(args):
    model_path = Path(args.model_path)
    if not model_path.exists():
        print(f"Model path {model_path} not found.")
        sys.exit(1)
        
    try:
        model = CheckpointManager.load_checkpoint(model_path)
        summary = summarize_model(model)
        print(json.dumps(summary, indent=2))
    except Exception as e:
        print(f"Failed to load model: {e}")
        sys.exit(1)

def cmd_validate(args):
    model_path = Path(args.model_path)
    try:
        print("Validating model configuration and weights...")
        CheckpointManager.load_checkpoint(model_path)
        print(f"Model {model_path} is structurally valid.")
    except Exception as e:
        print(f"Validation failed: {e}")

def main():
    parser = argparse.ArgumentParser(description="Vajra Model Management CLI")
    subparsers = parser.add_subparsers(dest="command", required=True)
    
    create_parser = subparsers.add_parser("create")
    create_parser.add_argument("--preset", default="Vajra-370M", help="Model preset name")
    create_parser.add_argument("--output-dir", required=True)
    create_parser.add_argument("--no-safetensors", dest="use_safetensors", action="store_false")
    create_parser.set_defaults(use_safetensors=True)
    
    summary_parser = subparsers.add_parser("summary")
    summary_parser.add_argument("model_path")
    
    validate_parser = subparsers.add_parser("validate")
    validate_parser.add_argument("model_path")
    
    args = parser.parse_args()
    
    if args.command == "create":
        cmd_create(args)
    elif args.command == "summary":
        cmd_summary(args)
    elif args.command == "validate":
        cmd_validate(args)

if __name__ == "__main__":
    main()
