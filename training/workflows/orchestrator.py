import logging
from typing import Optional
from tokenizer.tokenizers.hf_bpe import HFBpeTokenizer

from model.modeling import VajraForCausalLM
from training.production.engine import ProductionTrainingEngine
from training.ddp.dataloader import create_distributed_dataloader
from training.checkpoints.manager import TrainingCheckpointManager
from training.workflows.generation import TextGenerationPipeline
from training.workflows.reporting import TrainingReportGenerator
from training.workflows.preset import get_vajra_370m_preset, get_vajra_tiny_preset

logger = logging.getLogger(__name__)

class TrainingSessionManager:
    """Orchestrates the entire production training lifecycle."""
    
    def __init__(self, output_dir: str, dataset_dir: str, preset: str = "vajra-370m"):
        self.output_dir = output_dir
        self.dataset_dir = dataset_dir
        
        if preset == "vajra-tiny":
            model_cfg, prod_cfg, ddp_cfg = get_vajra_tiny_preset(output_dir, dataset_dir)
        else:
            model_cfg, prod_cfg, ddp_cfg = get_vajra_370m_preset(output_dir, dataset_dir)
            
        self.config = prod_cfg
        
        # Initialize bare components
        self.model = VajraForCausalLM(model_cfg)
        self.engine = ProductionTrainingEngine(self.model, self.config)
        
        # Note: in a real distributed environment DDP wrappers would be applied here,
        # but for this milestone we reuse the abstractions to build the workflow.
        
        # Utilities
        self.checkpoint_manager = TrainingCheckpointManager(self.output_dir, self.config.save_total_limit)
        
        # Mock tokenizer for the workflow integration
        self.tokenizer = None
        self.generator = TextGenerationPipeline(self.engine.model, self.tokenizer, self.engine.device)
        self.reporter = TrainingReportGenerator(self.output_dir)
        
        self.metrics_history = []
        
    def train(self, resume_checkpoint: Optional[str] = None):
        """Executes the training loop."""
        
        start_step = 0
        if resume_checkpoint:
            logger.info(f"Resuming from {resume_checkpoint}")
            progress = self.checkpoint_manager.load_checkpoint(
                resume_checkpoint, 
                self.engine.model, 
                self.engine.optimizer, 
                None # scheduler
            )
            start_step = progress.get("step", 0) if progress else 0
            
        dataloader = create_distributed_dataloader(
            self.dataset_dir,
            self.config.batch_size,
            self.config.max_sequence_length,
            rank=0,
            world_size=1,
            epoch=0
        )
        data_iter = iter(dataloader)
        
        self.engine.model.train()
        
        for step in range(start_step, self.config.max_steps):
            try:
                batch = next(data_iter)
            except StopIteration:
                data_iter = iter(dataloader)
                batch = next(data_iter)
                
            batch = batch.to(self.engine.device)
            metrics = self.engine.train_step(batch)
            self.metrics_history.append(metrics)
            
            if (step + 1) % getattr(self.config, "logging_steps", 10) == 0:
                loss = metrics.get("loss", float('nan'))
                print(f"Step {step+1}/{self.config.max_steps} - Loss: {loss:.4f}")
                
            if (step + 1) % getattr(self.config, "eval_steps", 1000) == 0:
                self.evaluate_and_report(step + 1)
                
            if (step + 1) % getattr(self.config, "save_steps", 1000) == 0:
                class DummyScheduler:
                    def state_dict(self): return {}
                self.checkpoint_manager.save_checkpoint(
                    step + 1,
                    self.engine.model,
                    self.engine.optimizer,
                    DummyScheduler(), # scheduler
                    metrics
                )
                
    def evaluate_and_report(self, step: int):
        """Runs evaluation and generates samples and reports."""
        prompts = ["The capital of France is", "Once upon a time in"]
        samples = self.generator.generate_samples(prompts, max_new_tokens=20)
        self.reporter.generate(step, self.metrics_history, samples)
        print(f"Generated report and samples for step {step}")
