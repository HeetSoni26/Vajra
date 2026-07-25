"""Publication script for uploading Vajra model release packages to Hugging Face Hub."""

import os
import sys
from pathlib import Path

try:
    from huggingface_hub import HfApi, create_repo
except ImportError:
    print("Error: huggingface_hub package is not installed. Install via `pip install huggingface_hub`.")
    sys.exit(1)


def publish_model(package_dir: str | Path, repo_id: str = "HeetSoni26/vajra-57m") -> None:
    try:
        import dotenv
        dotenv.load_dotenv(os.path.expanduser("~/.env"))
        dotenv.load_dotenv(".env")
    except ImportError:
        pass

    token = os.environ.get("HF_TOKEN")
    if not token:
        print("HF_TOKEN environment variable is missing. Attempting anonymous or pre-authenticated upload...")

    package_path = Path(package_dir)
    if not package_path.exists():
        raise FileNotFoundError(f"Package directory not found: {package_path}")

    print(f"Initializing Hugging Face API for repository: {repo_id}")
    api = HfApi(token=token)

    try:
        print(f"Ensuring repository '{repo_id}' exists...")
        create_repo(repo_id=repo_id, repo_type="model", private=False, exist_ok=True, token=token)
        print(f"Repository '{repo_id}' is ready.")
    except Exception as e:
        print(f"Repository creation warning/info: {e}")

    print(f"Uploading files from '{package_path}' to '{repo_id}'...")
    api.upload_folder(
        folder_path=str(package_path),
        repo_id=repo_id,
        repo_type="model",
        commit_message="feat(release): publish Vajra-57M v1.0.0 production release package",
        token=token,
    )
    print(f"Successfully uploaded package to https://huggingface.co/{repo_id}")


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Publish Vajra release package to Hugging Face Hub.")
    parser.add_argument("--package-dir", default="release/vajra-57m")
    parser.add_argument("--repo-id", default="HeetSoni26/vajra-57m")
    args = parser.parse_args()

    publish_model(args.package_dir, args.repo_id)


if __name__ == "__main__":
    main()
