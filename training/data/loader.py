from training.data_loader import create_dataloaders, MemmapTokenDataset
from pathlib import Path
from typing import Union
from torch.utils.data import DataLoader


def create_dataloader(
    data_dir: Union[str, Path],
    batch_size: int = 1,
    sequence_length: int = 2048,
    num_workers: int = 0,
) -> DataLoader:
    """Create train DataLoader from binary memmap files."""
    train_loader, _ = create_dataloaders(
        data_dir=data_dir,
        sequence_length=sequence_length,
        micro_batch_size=batch_size,
        num_workers=num_workers,
    )
    return train_loader


__all__ = ["create_dataloader", "create_dataloaders", "MemmapTokenDataset"]
