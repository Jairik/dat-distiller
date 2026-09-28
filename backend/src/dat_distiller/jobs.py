"""Background jobs: SQLite-persisted, SSE progress, cancellation, checkpoints.

Long-running work (Generation, Labeling, tuning) runs on a worker thread
while clients follow progress over SSE (see ``dat_distiller.api.jobs``).
Job types are registered with :meth:`JobManager.register`; a job type may
checkpoint partial results (``ctx.save_checkpoint``) so a cancelled or
interrupted run resumes with finished work skipped.

Statuses: queued → running → completed | failed | cancelled.
A server restart marks still-running rows ``interrupted``; resumable types
can then be resumed, the rest are marked failed.
"""

from __future__ import annotations

import json
import sqlite3
import threading
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Callable

TERMINAL_STATUSES = ("completed", "failed", "cancelled")
RETRYABLE_STATUSES = ("failed", "cancelled", "interrupted")

_SCHEMA = """
CREATE TABLE IF NOT EXISTS jobs (
    id TEXT PRIMARY KEY,
    type TEXT NOT NULL,
    project_id TEXT,
    status TEXT NOT NULL,
    resumable INTEGER NOT NULL DEFAULT 0,
    params_json TEXT NOT NULL DEFAULT '{}',
    progress_json TEXT NOT NULL DEFAULT '{}',
    checkpoint_json TEXT,
    result_json TEXT,
    error TEXT,
    created_at TEXT NOT NULL,
    started_at TEXT,
    finished_at TEXT
);

CREATE INDEX IF NOT EXISTS idx_jobs_project ON jobs(project_id);
"""


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


class JobNotFoundError(KeyError):
    pass


class JobStateError(ValueError):
    """Resume attempted on a job that cannot be resumed."""


@dataclass
class Job:
    id: str
    type: str
    project_id: str | None
    status: str
    resumable: bool
    params: dict[str, Any]
    progress: dict[str, Any]
    checkpoint: dict[str, Any] | None
    result: dict[str, Any] | None
    error: str | None
    created_at: str
    started_at: str | None
    finished_at: str | None

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "type": self.type,
            "project_id": self.project_id,
            "status": self.status,
            "resumable": bool(self.resumable),
            "params": self.params,
            "progress": self.progress,
            "result": self.result,
            "error": self.error,
            "created_at": self.created_at,
            "started_at": self.started_at,
            "finished_at": self.finished_at,
        }


class JobContext:
    """The runner's handle: progress reporting, cancellation, checkpoints."""

    def __init__(self, manager: "JobManager", job_id: str, cancel: threading.Event) -> None:
        self._manager = manager
        self.job_id = job_id
        self._cancel = cancel

    def progress(self, done: int, total: int, **extra: Any) -> None:
        """Persist progress; SSE subscribers pick it up within their poll interval."""
        self._manager._set_progress(self.job_id, {"done": done, "total": total, **extra})

    @property
    def cancelled(self) -> bool:
        """Set when the user cancels — cooperative runners check and stop."""
        return self._cancel.is_set()

    def save_checkpoint(self, data: dict[str, Any]) -> None:
        """Persist partial results so a resumed run can skip finished work."""
        self._manager._set_checkpoint(self.job_id, data)

    def load_checkpoint(self) -> dict[str, Any] | None:
        return self._manager.get(self.job_id).checkpoint


Runner = Callable[[JobContext], "dict[str, Any] | None"]


class JobManager:
    """Registry + lifecycle for background jobs. One per app (single user)."""

    def __init__(self, database) -> None:
        self.db = database
        self._runners: dict[str, tuple[Runner, bool]] = {}
        self._threads: dict[str, threading.Thread] = {}
        self._cancel_events: dict[str, threading.Event] = {}
        with self.db.connect() as conn:
            conn.executescript(_SCHEMA)
        self.recover()

    # -- registration --------------------------------------------------------

    def register(self, job_type: str, runner: Runner, *, resumable: bool = False) -> None:
        """Map a job type to ``runner(ctx)``; the return value is the result.

        A resumable runner MUST tolerate being called again with its saved
        checkpoint (``ctx.load_checkpoint()``) and skip finished work.
        """
        self._runners[job_type] = (runner, resumable)

    # -- lifecycle -----------------------------------------------------------

    def start(
        self,
        job_type: str,
        *,
        project_id: str | None = None,
        params: dict[str, Any] | None = None,
    ) -> Job:
        if job_type not in self._runners:
            raise ValueError(f"unknown job type {job_type!r}")
        runner, resumable = self._runners[job_type]
        job_id = uuid.uuid4().hex
        with self.db.connect() as conn:
            conn.execute(
                "INSERT INTO jobs (id, type, project_id, status, resumable, "
                "params_json, progress_json, created_at) "
                "VALUES (?, ?, ?, 'queued', ?, ?, '{}', ?)",
                (job_id, job_type, project_id, int(resumable), json.dumps(params or {}), _now()),
            )
        self._launch(job_id, runner)
        return self.get(job_id)

    def resume(self, job_id: str) -> Job:
        """Re-run a failed/cancelled/interrupted resumable job from its checkpoint."""
        job = self.get(job_id)
        if not job.resumable:
            raise JobStateError("this job type cannot be resumed")
        if job.status not in RETRYABLE_STATUSES:
            raise JobStateError(f"job is {job.status}; nothing to resume")
        runner = self._runners.get(job.type, (None, False))[0]
        if runner is None:
            raise JobStateError(f"job type {job.type!r} is not registered")
        with self.db.connect() as conn:
            conn.execute(
                "UPDATE jobs SET status = 'queued', error = NULL, finished_at = NULL WHERE id = ?",
                (job_id,),
            )
        self._launch(job_id, runner)
        return self.get(job_id)

    def cancel(self, job_id: str) -> Job:
        """Mark cancelled and signal the worker; cooperative runners stop fast."""
        job = self.get(job_id)
        if job.status in TERMINAL_STATUSES:
            return job
        with self.db.connect() as conn:
            conn.execute(
                "UPDATE jobs SET status = 'cancelled', finished_at = ? WHERE id = ?",
                (_now(), job_id),
            )
        event = self._cancel_events.get(job_id)
        if event is not None:
            event.set()
        return self.get(job_id)

    # -- running ---------------------------------------------------------------

    def _launch(self, job_id: str, runner: Runner) -> None:
        cancel = threading.Event()
        self._cancel_events[job_id] = cancel

        def work() -> None:
            with self.db.connect() as conn:
                conn.execute(
                    "UPDATE jobs SET status = 'running', started_at = ? WHERE id = ?",
                    (_now(), job_id),
                )
            ctx = JobContext(self, job_id, cancel)
            try:
                result = runner(ctx)
            except Exception as exc:  # noqa: BLE001 - failures are data, not crashes
                self._finish(job_id, "failed", error=f"{type(exc).__name__}: {exc}")
                return
            if cancel.is_set():
                self._finish(job_id, "cancelled")
            else:
                self._finish(job_id, "completed", result=result)

        thread = threading.Thread(target=work, name=f"job-{job_id[:8]}", daemon=True)
        self._threads[job_id] = thread
        thread.start()

    def _finish(
        self, job_id: str, status: str, *, result: dict | None = None, error: str | None = None
    ) -> None:
        # Never clobber a 'cancelled' status that cancel() already wrote.
        with self.db.connect() as conn:
            (current,) = conn.execute("SELECT status FROM jobs WHERE id = ?", (job_id,)).fetchone()
            if current == "cancelled":
                status = "cancelled"
            conn.execute(
                "UPDATE jobs SET status = ?, result_json = ?, error = ?, finished_at = ? WHERE id = ?",
                (status, json.dumps(result) if result is not None else None, error, _now(), job_id),
            )

    def _set_progress(self, job_id: str, progress: dict) -> None:
        with self.db.connect() as conn:
            conn.execute(
                "UPDATE jobs SET progress_json = ? WHERE id = ?", (json.dumps(progress), job_id)
            )

    def _set_checkpoint(self, job_id: str, data: dict) -> None:
        with self.db.connect() as conn:
            conn.execute(
                "UPDATE jobs SET checkpoint_json = ? WHERE id = ?", (json.dumps(data), job_id)
            )

    def recover(self) -> None:
        """After a server restart: resumable rows → interrupted, others → failed."""
        with self.db.connect() as conn:
            conn.execute(
                "UPDATE jobs SET status = 'interrupted', finished_at = ? "
                "WHERE status IN ('queued', 'running') AND resumable = 1",
                (_now(),),
            )
            conn.execute(
                "UPDATE jobs SET status = 'failed', error = 'interrupted by server restart', "
                "finished_at = ? WHERE status IN ('queued', 'running') AND resumable = 0",
                (_now(),),
            )

    # -- queries ---------------------------------------------------------------

    def get(self, job_id: str) -> Job:
        with self.db.connect() as conn:
            row = conn.execute("SELECT * FROM jobs WHERE id = ?", (job_id,)).fetchone()
        if row is None:
            raise JobNotFoundError(job_id)
        return self._from_row(row)

    def list_for_project(self, project_id: str) -> list[Job]:
        with self.db.connect() as conn:
            rows = conn.execute(
                "SELECT * FROM jobs WHERE project_id = ? ORDER BY created_at, rowid",
                (project_id,),
            ).fetchall()
        return [self._from_row(row) for row in rows]

    def list_jobs(self, limit: int = 100) -> list[Job]:
        """Most recent jobs across all projects (for the Running Jobs bar)."""
        with self.db.connect() as conn:
            rows = conn.execute(
                "SELECT * FROM jobs ORDER BY created_at DESC, rowid DESC LIMIT ?",
                (limit,),
            ).fetchall()
        return [self._from_row(row) for row in rows]

    @staticmethod
    def _from_row(row: sqlite3.Row) -> Job:
        return Job(
            id=row["id"],
            type=row["type"],
            project_id=row["project_id"],
            status=row["status"],
            resumable=bool(row["resumable"]),
            params=json.loads(row["params_json"] or "{}"),
            progress=json.loads(row["progress_json"] or "{}"),
            checkpoint=json.loads(row["checkpoint_json"]) if row["checkpoint_json"] else None,
            result=json.loads(row["result_json"]) if row["result_json"] else None,
            error=row["error"],
            created_at=row["created_at"],
            started_at=row["started_at"],
            finished_at=row["finished_at"],
        )
