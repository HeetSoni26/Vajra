import argparse
import torch
import torch.multiprocessing as mp

from model.config import get_preset
from model.modeling import VajraForCausalLM
from training.config import TrainingConfig
from training.ddp.config import DDPConfig
from training.ddp.init import init_process_group
from training.ddp.engine import DDPTrainingEngine


def _worker(rank: int, world_size: int, config: TrainingConfig, ddp_config: DDPConfig):
    """Entry point for each spawned DDP worker process."""
    init_process_group(ddp_config, rank=rank, world_size=world_size)

    model_config = get_preset("Vajra-370M")
    model = VajraForCausalLM(model_config)

    engine = DDPTrainingEngine(model, config, ddp_config, rank=rank, world_size=world_size)
    engine.train()


def cmd_train_ddp(args):
    train_config = TrainingConfig(
        dataset_dir=args.dataset_dir,
        output_dir=args.output_dir,
        batch_size=args.batch_size,
        max_steps=args.max_steps,
        mixed_precision=args.mixed_precision,
        device="cuda" if torch.cuda.is_available() else "cpu",
    )
    if args.config_file:
        train_config = TrainingConfig.load(args.config_file)

    ddp_config = DDPConfig(
        enabled=True,
        backend="nccl" if torch.cuda.is_available() else "gloo",
        master_addr=args.master_addr,
        master_port=args.master_port,
    )

    world_size = args.num_gpus or torch.cuda.device_count()
    if world_size < 1:
        world_size = 1

    print(f"Launching DDP training across {world_size} process(es).")
    mp.spawn(
        _worker,
        args=(world_size, train_config, ddp_config),
        nprocs=world_size,
        join=True,
    )


def cmd_dry_run(args):
    print("DDP dry-run: verifying configuration and imports...")
    ddp_config = DDPConfig(enabled=True, backend="gloo")
    print(
        f"DDPConfig: backend={ddp_config.backend}, master={ddp_config.master_addr}:{ddp_config.master_port}"
    )
    print("All DDP components imported successfully.")


def main():
    parser = argparse.ArgumentParser(description="Vajra DDP Training CLI")
    subparsers = parser.add_subparsers(dest="command", required=True)

    train_p = subparsers.add_parser("train-ddp")
    train_p.add_argument("--dataset-dir", required=True)
    train_p.add_argument("--output-dir", required=True)
    train_p.add_argument("--config-file")
    train_p.add_argument("--num-gpus", type=int, default=0)
    train_p.add_argument("--master-addr", default="127.0.0.1")
    train_p.add_argument("--master-port", type=int, default=29500)
    train_p.add_argument("--batch-size", type=int, default=8)
    train_p.add_argument("--max-steps", type=int, default=10000)
    train_p.add_argument("--mixed-precision", choices=["none", "fp16", "bf16"], default="bf16")

    subparsers.add_parser("dry-run")

    args = parser.parse_args()

    if args.command == "train-ddp":
        cmd_train_ddp(args)
    elif args.command == "dry-run":
        cmd_dry_run(args)


if __name__ == "__main__":
    main()
