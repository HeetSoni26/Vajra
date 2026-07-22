"""Background JobManager demonstration."""

from vajra_agent import JobManager


def main():
    jm = JobManager()
    job = jm.submit_job(
        task_description="Long-running model training pipeline",
        fn=lambda: "Model training finished successfully.",
    )

    print("Background Job Status Summary:")
    print(f"  ID: {job.id}")
    print(f"  Status: {job.status}")
    print(f"  Progress: {job.progress_pct}%")
    print(f"  Output: {job.output}")


if __name__ == "__main__":
    main()
