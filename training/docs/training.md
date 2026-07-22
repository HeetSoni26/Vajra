# Vajra Training Engine

The Vajra Single-GPU Training Engine defines robust iteration mappings structurally processing physical PyTorch boundaries seamlessly through native `.bin` dataset shards effectively.

## Components

- **`config.py` (`TrainingConfig`)**: Defines the global parameter mappings structurally setting hyper-parameters. Includes epochs, iterations, gradient bounds, logging limits, and checkpoint rotations natively.
- **`data/loader.py` (`BinaryShardDataset`)**: Creates logical memory boundaries actively mapping `mmap` views streaming large sequence boundaries mathematically cleanly spanning zero-copy allocations mapped actively.
- **`optim/optimizers.py`**: Isolates bias variables alongside standard layer normalization arrays mathematically omitting them implicitly out from AdamW's standard weight decay matrix mapping. 
- **`optim/schedulers.py`**: Exports mathematical mapping trajectories structurally processing Linear, Cosine, Step decay logic generating warm-up vectors naturally mapping back inside bounded configurations. 
- **`metrics/tracker.py`**: Accumulates tracking vectors safely spanning `.csv`, `.jsonl`, and TensorBoard endpoints implicitly mapping generation variables naturally preventing physical memory loss.
- **`checkpoints/manager.py`**: Encapsulates states natively capturing standard LLM definitions structurally wrapping RNG vectors safely generating rotation intervals tracking `save_total_limit`.
- **`engine/loop.py` (`TrainingEngine`)**: Fuses the data loaders sequentially directly against AMP Mixed Precision structures safely running accumulation arrays matching step variables tracking clipping parameters seamlessly without manual loops.

## CLI Utility

Operate physically bounding states straight out of `manage_training.py`:

```bash
# Execute Dry Run mapping data shapes logically ensuring hardware checks cleanly
python training/scripts/manage_training.py dry-run --dataset-dir output/shards

# Train physically tracking bounding topologies natively scaling bounds mapping dynamically.
python training/scripts/manage_training.py train --preset Vajra-370M --dataset-dir output/shards --output-dir output/training
```
