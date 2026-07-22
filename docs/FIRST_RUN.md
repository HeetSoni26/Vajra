# Vajra - First Run Guide

Welcome to Vajra. Follow this step-by-step guide to run your first training session.

1. **Verify Environment**: Ensure `uv` is installed and you are running Python 3.11+.
2. **Install Dependencies**: `uv pip install -r requirements.txt` (or install PyTorch 2.x manually based on CUDA version).
3. **Run Smoke Test**: `python scripts/smoke_test.py`. This runs a 10-second end-to-end check of the whole repo.
4. **Prepare Data**: Ensure your tokenized binary shards are inside `data/tokenized/`. 
5. **Pre-flight Checks**: `python scripts/preflight.py`
6. **Start Training**: `python training/workflows/scripts/launch.py train-370m --dataset-dir data/tokenized --output-dir checkpoints`
7. **View Dashboard**: Run `python scripts/dashboard.py` periodically or set up a cron job.
