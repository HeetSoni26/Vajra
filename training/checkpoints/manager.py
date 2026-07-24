import json
import shutil
from pathlib import Path
from typing import Any

import torch

from model.checkpoints import CheckpointManager as ModelCheckpointManager
from model.modeling import VajraForCausalLM


class TrainingCheckpointManager:
    """
    Wraps the Model CheckpointManager to additionally save and load:
    - Optimizer state
    - Scheduler state
    - RNG states
    - Training metrics/progress
    Also handles rotating older checkpoints to respect save_total_limit.
    """

    def __init__(self, output_dir: str | Path, save_total_limit: int = 3):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.save_total_limit = save_total_limit

    def save_checkpoint(
        self,
        step: int,
        model: VajraForCausalLM,
        optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler.LRScheduler,
        metrics: dict[str, Any],
        is_best: bool = False,
    ):
        checkpoint_dir = self.output_dir / f"checkpoint-{step}"
        checkpoint_dir.mkdir(parents=True, exist_ok=True)

        # 1. Save model weights & config
        ModelCheckpointManager.save_checkpoint(model, checkpoint_dir, use_safetensors=True)

        # 2. Save Optimizer and Scheduler states
        torch.save(optimizer.state_dict(), checkpoint_dir / "optimizer.pt")
        torch.save(scheduler.state_dict(), checkpoint_dir / "scheduler.pt")

        # 3. Save RNG state
        rng_state = {
            "torch_cpu": torch.get_rng_state(),
            "torch_cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
        }
        torch.save(rng_state, checkpoint_dir / "rng_state.pt")

        # 4. Save Training Progress
        progress = {"step": step, **metrics}
        with open(checkpoint_dir / "trainer_state.json", "w") as f:
            json.dump(progress, f, indent=2)

        # Rotate old checkpoints
        self._rotate_checkpoints()

        if is_best:
            best_dir = self.output_dir / "checkpoint-best"
            if best_dir.exists():
                shutil.rmtree(best_dir)
            shutil.copytree(checkpoint_dir, best_dir)

    def _rotate_checkpoints(self):
        checkpoints = [
            d
            for d in self.output_dir.iterdir()
            if d.is_dir() and d.name.startswith("checkpoint-") and d.name != "checkpoint-best"
        ]
        checkpoints = sorted(checkpoints, key=lambda x: int(x.name.split("-")[1]))

        if len(checkpoints) > self.save_total_limit:
            num_to_remove = len(checkpoints) - self.save_total_limit
            for d in checkpoints[:num_to_remove]:
                shutil.rmtree(d)

    def load_checkpoint(
        self,
        checkpoint_dir: str | Path,
        model: VajraForCausalLM,
        optimizer: torch.optim.Optimizer | None = None,
        scheduler: torch.optim.lr_scheduler.LRScheduler | None = None,
    ) -> dict[str, Any]:
        checkpoint_dir = Path(checkpoint_dir)

        # We assume the model weights were already loaded or we load them here?
        # Actually it's cleaner if we load weights into the model object directly here
        loaded_model = ModelCheckpointManager.load_checkpoint(
            checkpoint_dir, device=str(next(model.parameters()).device)
        )
        model.load_state_dict(loaded_model.state_dict())
        del loaded_model  # free mem

        if optimizer and (checkpoint_dir / "optimizer.pt").exists():
            optimizer.load_state_dict(
                torch.load(checkpoint_dir / "optimizer.pt", weights_only=True)
            )

        if scheduler and (checkpoint_dir / "scheduler.pt").exists():
            scheduler.load_state_dict(
                torch.load(checkpoint_dir / "scheduler.pt", weights_only=True)
            )

        if (checkpoint_dir / "rng_state.pt").exists():
            rng_state = torch.load(checkpoint_dir / "rng_state.pt", weights_only=True)
            torch.set_rng_state(rng_state["torch_cpu"])
            if rng_state["torch_cuda"] and torch.cuda.is_available():
                torch.cuda.set_rng_state_all(rng_state["torch_cuda"])

        if (checkpoint_dir / "trainer_state.json").exists():
            with open(checkpoint_dir / "trainer_state.json", "r") as f:
                return json.load(f)

        return {}
