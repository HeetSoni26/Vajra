import torch
from pathlib import Path

from model.checkpoints import CheckpointManager
from training.data.loader import create_dataloader
from evaluation.config import EvaluationConfig
from evaluation.metrics.standard import StandardMetrics
from evaluation.reporting.generator import ReportGenerator

class Evaluator:
    """
    Core Evaluation Engine. Loads a checkpoint, runs inference over a dataset,
    computes metrics, and generates a report.
    """
    def __init__(self, config: EvaluationConfig):
        self.config = config
        self.device = torch.device(config.device)
        self.dtype = torch.bfloat16 if config.mixed_precision == "bf16" else (torch.float16 if config.mixed_precision == "fp16" else torch.float32)
        
        self.metric_tracker = StandardMetrics()
        self.report_generator = ReportGenerator(config.output_dir)
        
    def evaluate(self, checkpoint_dir: str | Path, dataset_dir: str | Path = None) -> Path:
        checkpoint_dir = Path(checkpoint_dir)
        dataset_dir = Path(dataset_dir) if dataset_dir else Path(self.config.validation_datasets[0])
        
        print(f"Loading checkpoint from {checkpoint_dir}...")
        model = CheckpointManager.load_checkpoint(checkpoint_dir, device=self.config.device)
        model.eval()
        
        print(f"Initializing validation dataloader from {dataset_dir}...")
        dataloader = create_dataloader(
            str(dataset_dir),
            batch_size=self.config.batch_size,
            sequence_length=self.config.max_sequence_length
        )
        
        self.metric_tracker.reset()
        
        print("Starting evaluation...")
        with torch.no_grad():
            for batch in dataloader:
                batch = batch.to(self.device, non_blocking=True)
                
                with torch.autocast(device_type=self.device.type, dtype=self.dtype, enabled=self.config.mixed_precision != "none"):
                    outputs = model(input_ids=batch, labels=batch)
                    loss = outputs["loss"]
                    logits = outputs["logits"]
                    
                self.metric_tracker.update(logits, batch, loss)
                
        metrics = self.metric_tracker.compute()
        
        print("Evaluation complete. Generating report...")
        report_path = self.report_generator.generate(
            model_id=checkpoint_dir.name,
            metrics=metrics,
            hardware_info={"device": self.config.device, "precision": self.config.mixed_precision}
        )
        
        return report_path
