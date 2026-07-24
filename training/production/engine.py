import logging

import torch

from model.modeling import VajraForCausalLM
from training.production.config import ProductionConfig
from training.production.optimisation import optimise_model_for_production
from training.production.optimizer import create_production_optimizer
from training.production.profiler import MemoryProfiler, PerformanceProfiler
from training.production.watchdog import NumericalStabilityWatchdog

logger = logging.getLogger(__name__)


class ProductionTrainingEngine:
    """
    Production-grade training orchestrator supporting optimization,
    profiling, fault tolerance, and stability checks.
    """

    def __init__(self, model: VajraForCausalLM, config: ProductionConfig):
        self.config = config

        if torch.cuda.is_available():
            self.device = torch.device("cuda")
        else:
            self.device = torch.device("cpu")

        model = model.to(self.device)
        self.model = optimise_model_for_production(model, self.config.optimisation)

        self.optimizer = create_production_optimizer(
            self.model.parameters(), self.config, self.config.optimisation
        )

        self.watchdog = NumericalStabilityWatchdog(
            nan_detection=config.fault_tolerance.nan_detection,
            inf_detection=config.fault_tolerance.inf_detection,
        )

        self.mem_profiler = MemoryProfiler(self.device)
        self.perf_profiler = PerformanceProfiler()

    def train_step(self, batch: torch.Tensor) -> dict:
        """Executes a single production training step."""
        if self.config.profiling.enable_perf_profiling:
            self.perf_profiler.start_step()
            self.perf_profiler.start_forward()

        self.optimizer.zero_grad(set_to_none=True)

        outputs = self.model(input_ids=batch, labels=batch)
        loss = outputs["loss"]

        if self.config.profiling.enable_perf_profiling:
            self.perf_profiler.end_forward()
            self.perf_profiler.start_backward()

        # Watchdog check for loss stability
        if self.config.fault_tolerance.enable_watchdog and not self.watchdog.check_loss(
            loss.item()
        ):
            if self.config.fault_tolerance.skip_nan_gradients:
                logger.warning("Skipping step due to unstable loss.")
                return {"loss": float("nan")}

        loss.backward()

        if self.config.profiling.enable_perf_profiling:
            self.perf_profiler.end_backward()

        # Watchdog check for gradient stability
        if self.config.fault_tolerance.enable_watchdog and not self.watchdog.check_gradients(
            self.model
        ):
            if self.config.fault_tolerance.skip_nan_gradients:
                logger.warning("Skipping step due to unstable gradients.")
                self.optimizer.zero_grad(set_to_none=True)
                return {"loss": loss.item()}

        self.optimizer.step()

        metrics = {"loss": loss.item()}

        if self.config.profiling.enable_perf_profiling:
            metrics.update(self.perf_profiler.end_step())

        if self.config.profiling.enable_memory_profiling:
            metrics.update(self.mem_profiler.get_memory_stats())

        return metrics
