from pathlib import Path
from pydantic import BaseModel

class TrainingConfig(BaseModel):
    # Data params
    batch_size: int = 8
    max_sequence_length: int = 2048
    dataset_dir: str = "output/shards"
    
    # Optimization
    learning_rate: float = 3e-4
    weight_decay: float = 0.1
    adam_beta1: float = 0.9
    adam_beta2: float = 0.95
    adam_eps: float = 1e-8
    max_grad_norm: float = 1.0
    gradient_accumulation_steps: int = 1
    
    # Scheduler
    lr_scheduler_type: str = "cosine" # linear, cosine, constant, step
    warmup_steps: int = 100
    max_steps: int = 10000
    
    # Checkpointing
    output_dir: str = "output/training"
    save_steps: int = 1000
    save_total_limit: int = 3
    
    # Environment
    mixed_precision: str = "bf16" # none, fp16, bf16
    device: str = "cuda" # cuda, cpu
    seed: int = 42
    
    # Logging
    logging_steps: int = 10
    report_to: str = "none" # none, wandb, tensorboard
    
    def save(self, path: Path | str):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            f.write(self.model_dump_json(indent=2))
            
    @classmethod
    def load(cls, path: Path | str) -> 'TrainingConfig':
        with open(path, "r") as f:
            return cls.model_validate_json(f.read())
