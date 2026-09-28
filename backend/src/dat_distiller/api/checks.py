"""REST API for listing Checks and recording Acknowledgements."""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, HTTPException, Query, Request
from pydantic import BaseModel, Field

from ..checks import CheckNotFoundError, SUBJECT_TYPES

router = APIRouter(tags=["checks"])


class AcknowledgeIn(BaseModel):
    note: str | None = Field(default=None, max_length=2000)


@router.get("/checks")
def list_checks(
    request: Request,
    subject_type: str = Query(..., examples=["dataset_version"]),
    subject_id: str = Query(...),
    include_acknowledged: bool = Query(True),
) -> dict[str, Any]:
    """Checks for one subject plus the unacknowledged-warning count for the UI gate."""
    if subject_type not in SUBJECT_TYPES:
        raise HTTPException(422, f"subject_type must be one of {SUBJECT_TYPES}")
    store = request.app.state.checks
    checks = store.list_for_subject(
        subject_type, subject_id, include_acknowledged=include_acknowledged
    )
    return {
        "checks": [check.to_dict() for check in checks],
        "unacknowledged_warnings": len(store.unacknowledged_warnings(subject_type, subject_id)),
    }


@router.post("/checks/{check_id}/acknowledge")
def acknowledge_check(check_id: str, body: AcknowledgeIn, request: Request) -> dict[str, Any]:
    store = request.app.state.checks
    try:
        check = store.acknowledge(check_id, note=body.note)
    except CheckNotFoundError as exc:
        raise HTTPException(404, "check not found") from exc
    return check.to_dict()
