# Vajra Cloud Training & Multi-Provider Guide

## Overview

Vajra is a cloud-native training framework. Once `configs/training/cloud_sync.yaml` is configured, every checkpoint is automatically pushed to remote storage in the background. Training never pauses to wait for uploads.

On any new machine you only need to run:

```bash
python -m training.pretrain --resume
```

Vajra will:
1. Scan local experiment directories
2. If no valid local checkpoint is found, **automatically query the configured cloud provider**
3. Download the newest checkpoint
4. Validate it
5. Restore model, optimizer, scaler, and RNG state
6. Continue training from the exact step where it stopped

---

## Configuration (`configs/training/cloud_sync.yaml`)

| Key | Type | Description |
|-----|------|-------------|
| `enable_sync` | bool | Enable/disable cloud sync entirely |
| `provider` | str | `"huggingface"` or `"local"` |
| `repository` | str | HF Hub repo ID (e.g. `"user/vajra-checkpoints"`) |
| `private` | bool | Create/use a private HF repo |
| `upload_frequency` | int | Upload every N steps |
| `background_upload` | bool | Upload in background (non-blocking) |
| `retry_limit` | int | Max upload retries on failure |
| `retry_interval` | int | Seconds between retries |
| `download_on_resume` | bool | Auto-download from cloud if no local checkpoint |
| `compression` | bool | Reserved for future compression support |
| `metadata` | bool | Include metadata JSON in synced artifacts |

---

## Hugging Face Hub Setup

1. Create a HF account and generate a token at https://huggingface.co/settings/tokens
2. Export the token:
   ```bash
   export HF_TOKEN="hf_your_token_here"
   ```
3. Set `provider: huggingface` and `repository: "your-username/vajra-checkpoints"` in `cloud_sync.yaml`
4. Run training — Vajra will auto-create the repository if it does not exist

---

## Provider Portability Workflow

```
Local PC / Kaggle / Colab / Lambda / RunPod / Vast.ai
   │
   ▼
python -m training.pretrain --config configs/training/pretrain_tiny.yaml
   │
   ▼
CheckpointManager saves latest.pt every N steps
   │
   ▼
CloudSyncManager dispatches background upload to HF Hub
   │
   ▼
Training continues (non-blocking)
```

### Migrating Between Providers

1. Stop training on Provider A (or let it timeout)
2. Provision compute on Provider B
3. `git clone` the Vajra repository
4. `pip install -r requirements.txt`
5. Set `HF_TOKEN` environment variable
6. Run `python -m training.pretrain --resume`
7. Vajra automatically downloads and resumes — no manual file transfer needed

---

## Failure Recovery

| Failure Mode | Recovery |
|---|---|
| Network drop during upload | `retry_limit` automatic retries with `retry_interval` backoff |
| Corrupted local checkpoint | Falls back to older experiments, then queries cloud |
| Deleted local checkpoint | Cloud download activated automatically |
| Authentication failure | Clear error logged; manual token re-export required |
| Disk full | `CheckpointManager` disk guard trips; explicit error raised |
| Provider preemption | Resumes from last remote checkpoint on restart |

---

## Architecture

```
training/
├── cloud/
│   ├── __init__.py
│   ├── backends.py          # StorageBackend, HuggingFaceBackend, LocalBackend
│   └── sync_manager.py      # CloudSyncManager (orchestrator)
├── resume.py                # ResumeManager (local + cloud fallback)
├── checkpoint.py            # CheckpointManager (local save/load)
└── pretrain.py              # Entry point (clean, no cloud logic)
```

### Adding a New Provider

Implement the `StorageBackend` interface in `training/cloud/backends.py`:

```python
class MyNewBackend(StorageBackend):
    def upload_folder(self, local_dir, remote_path, run_as_future=True): ...
    def download_file(self, remote_path, local_dir): ...
    def list_files(self, remote_path): ...
```

Then register it in `CloudSyncManager._init_backend()`.
