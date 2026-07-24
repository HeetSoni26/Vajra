import numpy as np
import torch
from torch.utils.data import DataLoader, DistributedSampler


def create_distributed_dataloader(
    dataset_dir: str,
    batch_size: int,
    sequence_length: int,
    rank: int,
    world_size: int,
    epoch: int = 0,
    drop_last: bool = True,
    seed: int = 42,
) -> DataLoader:
    """
    Creates a DataLoader backed by a DistributedSampler.
    Each rank will receive a non-overlapping partition of the dataset.

    NOTE: BinaryShardDataset is an IterableDataset, so we cannot use
    DistributedSampler (which requires a map-style dataset).  Instead we
    wrap the raw numpy arrays in a thin map-style adapter so the sampler
    can partition them correctly.
    """
    dataset = _ShardMapDataset(dataset_dir, sequence_length)

    sampler = DistributedSampler(
        dataset,
        num_replicas=world_size,
        rank=rank,
        shuffle=True,
        seed=seed,
        drop_last=drop_last,
    )
    sampler.set_epoch(epoch)

    return DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=sampler,
        pin_memory=torch.cuda.is_available(),
        drop_last=drop_last,
    )


class _ShardMapDataset(torch.utils.data.Dataset):
    """
    Map-style view over binary shard sequences, required by DistributedSampler.
    Materialises the index on first access and caches it in memory.
    This is acceptable for research-scale training; for very large corpora,
    a chunked/streamed approach would be used.
    """

    def __init__(self, dataset_dir: str, sequence_length: int):
        from pathlib import Path

        self._seq_len = sequence_length
        meta_files = sorted(Path(dataset_dir).glob("*.json"))
        if not meta_files:
            raise ValueError(f"No shard metadata files found in {dataset_dir}")

        sequences = []
        from dataset.sharding.reader import BinaryShardReader

        for mf in meta_files:
            reader = BinaryShardReader(mf)
            for seq in reader.stream():
                sequences.append(seq.copy())

        self._data = sequences

    def __len__(self) -> int:
        return len(self._data)

    def __getitem__(self, idx: int) -> torch.Tensor:
        return torch.from_numpy(self._data[idx].astype(np.int64))
