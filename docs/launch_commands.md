# Vajra Training Launch Commands

## Single GPU Pretraining

**Vajra-Tiny**
```bash
python -m training.pretrain --config configs/training/pretrain_tiny.yaml
```

**Vajra-125M**
```bash
python -m training.pretrain --config configs/training/pretrain_125m.yaml
```

## Multi-GPU (DDP) Pretraining

Using `torchrun` to launch distributed data parallel training.

**2 GPUs (Vajra-125M)**
```bash
torchrun --nproc_per_node=2 -m training.pretrain --config configs/training/pretrain_125m.yaml
```

**4 GPUs (Vajra-370M)**
```bash
torchrun --nproc_per_node=4 -m training.pretrain --config configs/training/pretrain_370m.yaml
```

## Resuming from Checkpoint
Add the `--resume` flag to any launch command to automatically load the latest checkpoint and resume training safely (restores weights, optimizer, and RNG state).

```bash
python -m training.pretrain --config configs/training/pretrain_tiny.yaml --resume
```

## Supervised Fine-Tuning (SFT)
```bash
python -m training.sft --config configs/training/sft.yaml
```
