"""Job status, cancel/resume endpoints, and the SSE progress stream."""

from __future__ import annotations

import asyncio
import json
from typing import Any, AsyncIterator

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import StreamingResponse

from ..jobs import TERMINAL_STATUSES, JobManager, JobNotFoundError, JobStateError

router = APIRouter(tags=["jobs"])


def _manager(request: Request) -> JobManager:
    return request.app.state.jobs


@router.get("/jobs/{job_id}")
def get_job(job_id: str, request: Request) -> dict[str, Any]:
    try:
        return _manager(request).get(job_id).to_dict()
    except JobNotFoundError as exc:
        raise HTTPException(404, "job not found") from exc


@router.post("/jobs/{job_id}/cancel")
def cancel_job(job_id: str, request: Request) -> dict[str, Any]:
    try:
        return _manager(request).cancel(job_id).to_dict()
    except JobNotFoundError as exc:
        raise HTTPException(404, "job not found") from exc


@router.post("/jobs/{job_id}/resume")
def resume_job(job_id: str, request: Request) -> dict[str, Any]:
    try:
        return _manager(request).resume(job_id).to_dict()
    except JobNotFoundError as exc:
        raise HTTPException(404, "job not found") from exc
    except JobStateError as exc:
        raise HTTPException(409, str(exc)) from exc


@router.get("/jobs/{job_id}/events")
async def job_events(job_id: str, request: Request) -> StreamingResponse:
    """SSE: ``progress`` events while running, then the terminal status event
    (``completed`` / ``failed`` / ``cancelled`` / ``interrupted``), then ``end``.
    A page reload simply re-subscribes to the still-running job."""
    manager = _manager(request)
    try:
        manager.get(job_id)
    except JobNotFoundError as exc:
        raise HTTPException(404, "job not found") from exc

    async def stream() -> AsyncIterator[str]:
        last_sent: str | None = None
        while True:
            job = manager.get(job_id)
            data = json.dumps(
                {
                    "status": job.status,
                    "progress": job.progress,
                    "result": job.result,
                    "error": job.error,
                }
            )
            if data != last_sent:
                event = job.status if job.status in TERMINAL_STATUSES else "progress"
                yield f"event: {event}\ndata: {data}\n\n"
                last_sent = data
            if job.status in TERMINAL_STATUSES:
                yield "event: end\ndata: {}\n\n"
                return
            await asyncio.sleep(0.05)

    return StreamingResponse(
        stream(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )
