import argparse
import json
import math
import time
from pathlib import Path

import yaml

from model import FoundationLM, ModelConfig
from training.checkpoint import load_checkpoint
from training.data_loader import create_dataloaders
from training.trainer import Trainer
from utils.environment import get_device, get_git_hash
from utils.logging import setup_logger

logger = setup_logger("evaluate")


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate a Vajra checkpoint.")
    parser.add_argument("--checkpoint", required=True, help="Path to checkpoint .pt file or directory containing latest.pt")
    parser.add_argument("--config", required=True, help="Path to training config.yaml")
    parser.add_argument("--data-dir", default=None, help="Path to evaluation dataset (defaults to config data_dir)")
    parser.add_argument("--batch-size", type=int, default=1, help="Evaluation batch size")
    parser.add_argument("--sequence-length", type=int, default=None, help="Override sequence length")
    
    args = parser.parse_args()
    
    ckpt_path = Path(args.checkpoint)
    if ckpt_path.is_dir():
        ckpt_path = ckpt_path / "latest.pt"
        
    cfg_path = Path(args.config)
    cfg = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))
    
    # Setup Data
    data_dir = args.data_dir or cfg.get("data_dir", "data/tokenized")
    data_dir = Path(data_dir)
    
    # Load Model Config
    model_cfg_path = Path(cfg["model_config"])
    model_cfg = ModelConfig.from_yaml(model_cfg_path)
    model = FoundationLM(model_cfg)
    
    # Load Checkpoint
    logger.info(f"Loading checkpoint from: {ckpt_path}")
    state = load_checkpoint(ckpt_path, model)
    
    global_step = state.get("step", 0)
    tokens_seen = state.get("tokens_seen", 0)
    
    device = get_device()
    model = model.to(device)
    model.eval()
    
    # Load Validation DataLoader
    sequence_length = args.sequence_length or int(cfg.get("sequence_length", 2048))
    _, val_loader = create_dataloaders(
        data_dir=data_dir,
        sequence_length=sequence_length,
        micro_batch_size=args.batch_size,
        is_distributed=False,
        world_size=1,
        rank=0,
    )
    
    if val_loader is None:
        raise ValueError(f"No validation data found in {data_dir}. Ensure val.bin exists.")
    
    logger.info("Running evaluation...")
    start_time = time.time()
    
    # Setup dummy trainer for evaluation logic
    trainer = Trainer(
        model=model,
        optimizer=None,
        device=device,
        precision=str(cfg.get("precision", "fp32")),
    )
    
    val_stats = trainer.evaluate(val_loader)
    val_loss = val_stats.get("val_loss", 0.0)
    perplexity = math.exp(val_loss)
    eval_duration = time.time() - start_time
    
    # Save Results
    step_str = str(global_step)
    eval_dir = Path("evaluations") / f"checkpoint_{step_str}"
    eval_dir.mkdir(parents=True, exist_ok=True)
    
    results = {
        "checkpoint_filename": ckpt_path.name,
        "global_step": global_step,
        "tokens_seen": tokens_seen,
        "validation_loss": val_loss,
        "perplexity": perplexity,
        "evaluation_timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "model_name": getattr(model_cfg, "model_name", "vajra-lm"),
        "dataset_name": data_dir.name,
        "configuration_path": str(cfg_path),
        "git_commit_hash": get_git_hash(),
        "evaluation_duration_sec": eval_duration,
    }
    
    metrics_path = eval_dir / "metrics.json"
    with metrics_path.open("w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
        
    logger.info(f"Evaluation complete. Loss: {val_loss:.4f} | Perplexity: {perplexity:.4f}")
    logger.info(f"Results saved to {metrics_path}")


if __name__ == "__main__":
    main()
