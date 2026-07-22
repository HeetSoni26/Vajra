from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.distributed import DistributedSampler


class MemmapTokenDataset(Dataset):
    """High-throughput memory-mapped token dataset reading binary uint32 token files."""

    def __init__(self, path: str | Path, sequence_length: int) -> None:
        self.path = Path(path)
        if not self.path.exists():
            raise FileNotFoundError(f"Token binary file not found: {self.path}")
        self.tokens = np.memmap(self.path, dtype=np.uint32, mode="r")
        self.sequence_length = sequence_length

    def __len__(self) -> int:
        return max(0, (len(self.tokens) - 1) // self.sequence_length)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        start = idx * self.sequence_length
        end = start + self.sequence_length + 1
        chunk = self.tokens[start:end].astype(np.int64)

        if len(chunk) < self.sequence_length + 1:
            # Pad if at boundary
            pad_len = (self.sequence_length + 1) - len(chunk)
            chunk = np.pad(chunk, (0, pad_len), mode="constant", constant_values=0)

        return {
            "input_ids": torch.from_numpy(chunk[:-1].copy()),
            "labels": torch.from_numpy(chunk[1:].copy()),
        }


def create_dataloaders(
    data_dir: str | Path,
    sequence_length: int,
    micro_batch_size: int,
    num_workers: int = 0,
    is_distributed: bool = False,
    world_size: int = 1,
    rank: int = 0,
) -> tuple[DataLoader, DataLoader | None]:
    """Create train and validation PyTorch DataLoaders from binary memmap files with DistributedSampler support."""
    data_dir = Path(data_dir)

    train_path = data_dir / "train.bin"
    if not train_path.exists():
        # Fallback to tokens.bin if train.bin does not exist
        train_path = data_dir / "tokens.bin"

    train_dataset = MemmapTokenDataset(train_path, sequence_length)

    if is_distributed and world_size > 1:
        train_sampler = DistributedSampler(train_dataset, num_replicas=world_size, rank=rank, shuffle=True)
        train_loader = DataLoader(
            train_dataset,
            batch_size=micro_batch_size,
            sampler=train_sampler,
            num_workers=num_workers,
            pin_memory=torch.cuda.is_available(),
        )
    else:
        train_loader = DataLoader(
            train_dataset,
            batch_size=micro_batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=torch.cuda.is_available(),
        )

    val_loader: DataLoader | None = None
    val_path = data_dir / "val.bin"
    if val_path.exists() and val_path.stat().st_size > 0:
        val_dataset = MemmapTokenDataset(val_path, sequence_length)
        if is_distributed and world_size > 1:
            val_sampler = DistributedSampler(val_dataset, num_replicas=world_size, rank=rank, shuffle=False)
            val_loader = DataLoader(
                val_dataset,
                batch_size=micro_batch_size,
                sampler=val_sampler,
                num_workers=num_workers,
                pin_memory=torch.cuda.is_available(),
            )
        else:
            val_loader = DataLoader(
                val_dataset,
                batch_size=micro_batch_size,
                shuffle=False,  # Deterministic validation
                num_workers=num_workers,
                pin_memory=torch.cuda.is_available(),
            )

    return train_loader, val_loader
