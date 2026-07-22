"""Tests for JobManager long-running background tasks."""

from vajra_agent import JobManager, JobStatus


def test_job_manager_lifecycle():
    jm = JobManager()
    job = jm.submit_job("Background task execution", fn=lambda: "completed output")

    assert job.status == JobStatus.COMPLETED
    assert job.output == "completed output"
    assert job.progress_pct == 100.0
