from pathlib import Path

from torch.utils.data import DataLoader

from training.data_loader import MemmapTokenDataset, create_dataloaders


def create_dataloader(
    data_dir: str | Path,
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


__all__ = ["MemmapTokenDataset", "create_dataloader", "create_dataloaders"]
