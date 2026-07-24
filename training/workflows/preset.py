from model.config import VajraConfig
from training.production.config import (
    ProductionConfig,
    OptimisationConfig,
    FaultToleranceConfig,
    ProfilingConfig,
)
from training.ddp.config import DDPConfig


def get_vajra_370m_preset(
    output_dir: str, dataset_dir: str
) -> (VajraConfig, ProductionConfig, DDPConfig):
    """
    Production-ready preset configuration for training Vajra-370M.
    """
    model_config = VajraConfig(
        vocab_size=32000,
        hidden_size=1024,
        intermediate_size=4096,
        num_layers=24,
        num_attention_heads=16,
        num_key_value_heads=16,
        max_position_embeddings=4096,
        rms_norm_eps=1e-6,
    )

    train_config = ProductionConfig(
        dataset_dir=dataset_dir,
        output_dir=output_dir,
        batch_size=8,
        gradient_accumulation_steps=4,  # Effective batch = batch * accum * num_gpus
        learning_rate=3e-4,
        weight_decay=0.1,
        max_steps=100000,
        warmup_steps=2000,
        max_grad_norm=1.0,
        mixed_precision="bf16",
        max_sequence_length=4096,
        seed=42,
        save_steps=1000,
        logging_steps=10,
        eval_steps=1000,
        save_total_limit=5,
        optimisation=OptimisationConfig(
            gradient_checkpointing=True,
            compile_model=True,
            compile_backend="inductor",
            use_flash_attention=True,
            fused_optimizer=True,
        ),
        fault_tolerance=FaultToleranceConfig(
            enable_watchdog=True,
            nan_detection=True,
            inf_detection=True,
            skip_nan_gradients=True,
            max_retries=3,
            checkpoint_rotation=True,
            keep_best_checkpoints=3,
        ),
        profiling=ProfilingConfig(enable_memory_profiling=True, enable_perf_profiling=True),
    )

    ddp_config = DDPConfig(
        enabled=True,
        backend="nccl",
        find_unused_parameters=False,
        static_graph=True,
        broadcast_buffers=False,
    )

    return model_config, train_config, ddp_config


def get_vajra_tiny_preset(
    output_dir: str, dataset_dir: str
) -> (VajraConfig, ProductionConfig, DDPConfig):
    """
    Miniature synthetic configuration strictly for fast offline testing.
    """
    model_config = VajraConfig(
        vocab_size=1024,
        hidden_size=32,
        intermediate_size=64,
        num_layers=2,
        num_attention_heads=2,
        num_key_value_heads=2,
        max_position_embeddings=128,
    )

    train_config = ProductionConfig(
        dataset_dir=dataset_dir,
        output_dir=output_dir,
        batch_size=2,
        gradient_accumulation_steps=1,
        learning_rate=1e-3,
        max_steps=10,
        warmup_steps=2,
        mixed_precision="none",
        max_sequence_length=32,
        save_steps=5,
        logging_steps=2,
        eval_steps=5,
        save_total_limit=2,
        optimisation=OptimisationConfig(
            gradient_checkpointing=False,
            compile_model=False,
            use_flash_attention=False,
            fused_optimizer=False,
        ),
        fault_tolerance=FaultToleranceConfig(enable_watchdog=False),
        profiling=ProfilingConfig(enable_perf_profiling=False, enable_memory_profiling=False),
    )

    ddp_config = DDPConfig(enabled=False, backend="gloo")

    return model_config, train_config, ddp_config
