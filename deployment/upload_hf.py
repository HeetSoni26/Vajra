from __future__ import annotations

import argparse
from huggingface_hub import HfApi


def main() -> None:
    parser = argparse.ArgumentParser(description="Upload model folder to HuggingFace Hub.")
    parser.add_argument("--folder", default="checkpoints/final/hf")
    parser.add_argument("--repo_id", required=True)
    parser.add_argument("--private", action="store_true")
    args = parser.parse_args()
    api = HfApi()
    api.create_repo(args.repo_id, repo_type="model", private=args.private, exist_ok=True)
    api.upload_folder(folder_path=args.folder, repo_id=args.repo_id, repo_type="model")


if __name__ == "__main__":
    main()
