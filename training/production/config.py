from typing import List
from pydantic import BaseModel, Field

from training.config import TrainingConfig


class OptimisationConfig(BaseModel):
    """Configuration for production performance optimizations."""
    gradient_checkpointing: bool = False
    compile_model: bool = False
    compile_backend: str = "inductor"
    use_flash_attention: bool = False
    fused_optimizer: bool = True


class FaultToleranceConfig(BaseModel):
    """Configuration for numerical stability and fault recovery."""
    enable_watchdog: bool = True
    nan_detection: bool = True
    inf_detection: bool = True
    skip_nan_gradients: bool = True
    max_retries: int = 3
    checkpoint_rotation: bool = True
    keep_best_checkpoints: int = 3


class ProfilingConfig(BaseModel):
    """Configuration for memory and performance profiling."""
    enable_memory_profiling: bool = False
    enable_perf_profiling: bool = False
    profile_steps: List[int] = Field(default_factory=lambda: [10, 50, 100])
    profile_memory_interval: int = 100


class MultiNodeConfig(BaseModel):
    """
    Abstractions for future multi-node support.
    Currently inactive.
    """
    enabled: bool = False
    cluster_env: str = "torchrun"  # e.g., torchrun, slurm
    nodes: int = 1
    rendezvous_backend: str = "c10d"


class ProductionConfig(TrainingConfig):
    """
    Production-grade training configuration extending the base TrainingConfig.
    """
    optimisation: OptimisationConfig = Field(default_factory=OptimisationConfig)
    fault_tolerance: FaultToleranceConfig = Field(default_factory=FaultToleranceConfig)
    profiling: ProfilingConfig = Field(default_factory=ProfilingConfig)
    multi_node: MultiNodeConfig = Field(default_factory=MultiNodeConfig)
    eval_steps: int = 1000
