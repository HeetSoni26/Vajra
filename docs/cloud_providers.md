# Vajra Cloud Provider Support

This document provides instructions for launching Vajra training on popular cloud GPU providers.

## RunPod
RunPod offers cost-effective instances. Use the **PyTorch 2.x** template.

### Setup
1. Deploy a Pod (e.g., 1x RTX A6000 or 1x RTX 4090).
2. SSH into the pod or use the web terminal.
3. Clone Vajra:
   ```bash
   git clone https://github.com/HeetSoni26/Vajra.git
   cd Vajra
   ```
4. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
5. Synchronize dataset:
   ```bash
   # Use rsync or aws cli to pull tokenized data
   aws s3 sync s3://your-bucket/data/tokenized/ data/tokenized/
   ```
6. Launch training:
   ```bash
   bash scripts/launch_cloud_training.sh tiny
   ```

## Lambda Labs
Lambda Labs instances provide raw Ubuntu with NVIDIA drivers.

### Setup
1. Launch an instance (e.g., 1x A100 or 4x A100).
2. SSH into the instance.
3. Set up the environment:
   ```bash
   sudo apt update && sudo apt install -y python3.11-venv
   python3.11 -m venv venv
   source venv/bin/activate
   git clone https://github.com/HeetSoni26/Vajra.git
   cd Vajra
   pip install -r requirements.txt
   ```
4. Mount network storage or download dataset directly.
5. Launch distributed training (if 4x A100):
   ```bash
   torchrun --nproc_per_node=4 -m training.pretrain --config configs/training/pretrain_370m.yaml
   ```

## Checkpoint Synchronization & Recovery
To prevent data loss on preemptible instances:
- Set `save_every_steps` in your config to a frequent interval (e.g., every 30-60 minutes).
- Use a background cron job to sync `checkpoints/` to S3/GCS.
- On instance restart, sync `checkpoints/` back, and the launch script's `--resume` flag will automatically pick up from the latest `.pt` file.
