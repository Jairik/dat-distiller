"""Job framework: progress, SSE, cancellation, checkpointed resumption, pool."""

from __future__ import annotations

import threading
import time

import pytest
from fastapi.testclient import TestClient

from dat_distiller.jobs import JobManager, JobStateError
from dat_distiller.pool import run_pool


@pytest.fixture
def manager(isolated_data_dir) -> JobManager:
    from dat_distiller.store.db import Database
    from dat_distiller.store.paths import AppPaths

    paths = AppPaths(root=isolated_data_dir)
    return JobManager(Database(paths.db_path))


def wait_for_status(manager: JobManager, job_id: str, status: str, timeout: float = 5.0):
    deadline = time.time() + timeout
    while time.time() < deadline:
        job = manager.get(job_id)
        if job.status == status:
            return job
        time.sleep(0.02)
    raise AssertionError(f"job {job_id} never reached {status} (last: {manager.get(job_id).status})")


# -- manager ---------------------------------------------------------------


def test_completes_with_progress_and_result(manager: JobManager) -> None:
    def counting(ctx):
        for i in range(1, 6):
            ctx.progress(done=i, total=5)
        return {"answer": 42}

    manager.register("count", counting)
    job = manager.start("count", params=None)
    done = wait_for_status(manager, job.id, "completed")
    assert done.result == {"answer": 42}
    assert done.progress == {"done": 5, "total": 5}


def test_failure_is_captured_as_data(manager: JobManager) -> None:
    def boom(ctx):
        raise RuntimeError("kaboom")

    manager.register("boom", boom)
    job = manager.start("boom")
    failed = wait_for_status(manager, job.id, "failed")
    assert failed.error == "RuntimeError: kaboom"


def test_cancel_stops_promptly_and_keeps_status(manager: JobManager) -> None:
    started = threading.Event()

    def slow(ctx):
        started.set()
        while not ctx.cancelled:
            time.sleep(0.01)
        return None

    manager.register("slow", slow)
    job = manager.start("slow")
    assert started.wait(5)
    cancelled = manager.cancel(job.id)
    assert cancelled.status == "cancelled"
    # the worker finishing late must not flip the status back to completed
    time.sleep(0.1)
    assert manager.get(job.id).status == "cancelled"


def test_resumable_job_skips_checkpointed_work(manager: JobManager) -> None:
    processed: list[int] = []
    fail_on = {3}

    def chunked(ctx):
        done = (ctx.load_checkpoint() or {}).get("done", [])
        for item in range(5):
            if item in done:
                continue
            if item in fail_on:
                raise RuntimeError(f"item {item} exploded")
            processed.append(item)
            ctx.save_checkpoint({"done": done + [item]})
            done.append(item)
        return {"processed": processed}

    manager.register("chunked", chunked, resumable=True)
    job = manager.start("chunked")
    wait_for_status(manager, job.id, "failed")
    assert processed == [0, 1, 2]  # 3 killed the run; work up to it kept

    fail_on.clear()  # transient failure fixed
    resumed = manager.resume(job.id)
    assert resumed.id == job.id  # same row reused; it may already have finished
    completed = wait_for_status(manager, job.id, "completed")
    # resumed run appended only 3 and 4 — no duplicates means 0-2 were skipped
    assert completed.result["processed"] == [0, 1, 2, 3, 4]
    assert completed.checkpoint["done"] == [0, 1, 2, 3, 4]


def test_resume_rejects_unresumable_and_completed(manager: JobManager) -> None:
    manager.register("quick", lambda ctx: {})
    job = manager.start("quick")
    wait_for_status(manager, job.id, "completed")
    with pytest.raises(JobStateError):
        manager.resume(job.id)

    manager.register("resumable", lambda ctx: {}, resumable=True)
    done = manager.start("resumable")
    wait_for_status(manager, done.id, "completed")
    with pytest.raises(JobStateError):
        manager.resume(done.id)


def test_restart_recovery_marks_interrupted_or_failed(isolated_data_dir) -> None:
    from dat_distiller.store.db import Database
    from dat_distiller.store.paths import AppPaths

    paths = AppPaths(root=isolated_data_dir)
    db = Database(paths.db_path)
    first = JobManager(db)
    first.register("resumable", lambda ctx: {}, resumable=True)
    first.register("plain", lambda ctx: {})
    # simulate rows that were mid-flight when the server died
    with db.connect() as conn:
        conn.execute(
            "INSERT INTO jobs (id, type, status, resumable, params_json, progress_json, created_at) "
            "VALUES ('a', 'resumable', 'running', 1, '{}', '{}', 'x')"
        )
        conn.execute(
            "INSERT INTO jobs (id, type, status, resumable, params_json, progress_json, created_at) "
            "VALUES ('b', 'plain', 'running', 0, '{}', '{}', 'x')"
        )
    restarted = JobManager(db)
    assert restarted.get("a").status == "interrupted"
    assert restarted.get("b").status == "failed"
    # interrupted resumable jobs are resumable (runner is instant: may finish at once)
    restarted.register("resumable", lambda ctx: {}, resumable=True)
    restarted.resume("a")
    wait_for_status(restarted, "a", "completed")


# -- pool helper ------------------------------------------------------------


def test_pool_retries_with_backoff_then_reports_failures() -> None:
    slept: list[float] = []
    attempts: dict[int, int] = {}

    def flaky(item: int):
        attempts[item] = attempts.get(item, 0) + 1
        if item == 2 and attempts[item] < 3:
            raise ConnectionError("transient")
        if item == 3:
            raise ValueError("permanent")
        return item * 10

    outcome = run_pool(
        [1, 2, 3], flaky, concurrency=2, retries=2, backoff_base=0.5, sleep=slept.append
    )
    assert sorted(outcome.values) == [10, 20]
    assert [f.item for f in outcome.failures] == [3]
    assert outcome.failures[0].attempts == 3
    assert slept[0] == 0.5 and 1.0 in slept  # exponential backoff schedule


def test_pool_cancellation_skips_unstarted_items() -> None:
    outcome_holder: dict = {}
    cancel_flag = {"v": False}

    def slow_item(item: int) -> int:
        time.sleep(0.01)
        return item

    t = threading.Thread(
        target=lambda: outcome_holder.update(
            o=run_pool(
                list(range(50)),
                slow_item,
                concurrency=1,
                retries=0,
                should_cancel=lambda: cancel_flag["v"],
            )
        )
    )
    t.start()
    time.sleep(0.05)
    cancel_flag["v"] = True
    t.join(5)
    outcome = outcome_holder["o"]
    assert outcome.cancelled is True
    assert any(r.skipped for r in outcome.results)  # queued work never ran


# -- API + SSE ---------------------------------------------------------------


def test_job_endpoints_and_sse(client: TestClient) -> None:
    manager = client.app.state.jobs  # type: ignore[attr-defined]

    def counting(ctx):
        ctx.progress(done=1, total=3)
        time.sleep(0.25)  # give the SSE subscriber a chance to see progress
        ctx.progress(done=3, total=3)
        return {"ok": True}

    manager.register("count", counting)
    job = manager.start("count")

    events = client.get(f"/api/jobs/{job.id}/events").text
    assert "event: progress" in events
    assert "event: completed" in events
    assert '"done": 3' in events
    assert "event: end" in events

    fetched = client.get(f"/api/jobs/{job.id}").json()
    assert fetched["status"] == "completed" and fetched["result"] == {"ok": True}
    assert client.get("/api/jobs/missing").status_code == 404
    assert client.post("/api/jobs/missing/cancel").status_code == 404
    assert client.post("/api/jobs/missing/resume").status_code == 404

    manager.register("resumable-quick", lambda ctx: {}, resumable=True)
    quick = manager.start("resumable-quick")
    wait_for_status(manager, quick.id, "completed")
    assert client.post(f"/api/jobs/{quick.id}/resume").status_code == 409
