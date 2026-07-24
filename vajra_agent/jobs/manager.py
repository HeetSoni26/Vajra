"""JobManager managing background task execution, pause, resume, cancel, and checkpointing."""

from __future__ import annotations

import time
import uuid
from collections.abc import Callable
from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class JobStatus(str, Enum):
    """Job status lifecycle state."""

    PENDING = "pending"
    RUNNING = "running"
    PAUSED = "paused"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


@dataclass
class Job:
    """Represents a long-running background task job."""

    id: str
    task_description: str
    status: JobStatus = JobStatus.PENDING
    progress_pct: float = 0.0
    output: Any = None
    error: str | None = None
    checkpoints: list[dict[str, Any]] = field(default_factory=list)
    created_at: float = field(default_factory=time.time)


class JobManager:
    """Manages background jobs, pause, resume, cancel, progress, and checkpointing."""

    def __init__(self) -> None:
        self.jobs: dict[str, Job] = {}

    def submit_job(self, task_description: str, fn: Callable[[], Any] | None = None) -> Job:
        """Submit a new background job."""
        job_id = f"job_{uuid.uuid4().hex[:8]}"
        job = Job(id=job_id, task_description=task_description, status=JobStatus.RUNNING)
        self.jobs[job_id] = job

        if fn:
            try:
                out = fn()
                job.output = out
                job.status = JobStatus.COMPLETED
                job.progress_pct = 100.0
            except Exception as e:
                job.error = str(e)
                job.status = JobStatus.FAILED

        return job

    def pause_job(self, job_id: str) -> bool:
        """Pause a running job."""
        job = self.jobs.get(job_id)
        if job and job.status == JobStatus.RUNNING:
            job.status = JobStatus.PAUSED
            return True
        return False

    def resume_job(self, job_id: str) -> bool:
        """Resume a paused job."""
        job = self.jobs.get(job_id)
        if job and job.status == JobStatus.PAUSED:
            job.status = JobStatus.RUNNING
            return True
        return False

    def cancel_job(self, job_id: str) -> bool:
        """Cancel a running or paused job."""
        job = self.jobs.get(job_id)
        if job and job.status in (JobStatus.RUNNING, JobStatus.PAUSED, JobStatus.PENDING):
            job.status = JobStatus.CANCELLED
            return True
        return False

    def checkpoint_job(self, job_id: str, state_data: dict[str, Any]) -> bool:
        """Save a execution checkpoint for a job."""
        job = self.jobs.get(job_id)
        if job:
            job.checkpoints.append({"timestamp": time.time(), "state": state_data})
            return True
        return False

    def get_job_status(self, job_id: str) -> Job | None:
        """Retrieve job by ID."""
        return self.jobs.get(job_id)
