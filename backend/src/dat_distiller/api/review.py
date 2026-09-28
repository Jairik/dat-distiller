"""Review Queue endpoints: read the queue, apply decisions, check the remainder.

Three calls, and each answers a different question:

- ``GET  /dataset-versions/{id}/review-queue`` — what is waiting, and at which
  threshold. Read-only; it registers nothing.
- ``POST /dataset-versions/{id}/review`` — apply decisions. Produces a **child**
  Dataset Version, because the one you reviewed is immutable.
- ``GET  /dataset-versions/{id}/review-status`` — just the count, which is what
  the Train step polls; it deliberately does not carry the queue itself.
"""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, HTTPException, Query, Request
from pydantic import BaseModel, Field

from ..review import (
    DECISIONS,
    UNREVIEWED_CHECK_KIND,
    apply_review,
    build_queue,
    label_families,
    register_unreviewed_check,
    validate_decisions,
)

router = APIRouter(tags=["review"])


class ReviewDecisionIn(BaseModel):
    """One row's decision. `override` is required only for an override."""

    row_index: int = Field(ge=0)
    family: str = Field(min_length=1)
    question_type: str = "noul"
    decision: str
    override: Any = None
    note: str | None = Field(default=None, max_length=2000)


class ReviewIn(BaseModel):
    decisions: list[ReviewDecisionIn] = Field(default_factory=list)
    threshold: float | None = Field(default=None, ge=0, le=1)
    raise_check: bool = True


def _threshold(request: Request, override: float | None) -> float:
    """The configured review threshold unless the caller names one."""
    if override is not None:
        return override
    return float(request.app.state.settings.load().get("review_threshold", 0.8))


def _queue_for(request: Request, version_id: str, threshold: float, limit: int | None):
    store = request.app.state.store
    frame = store.load_dataframe(version_id, include_provenance=True)
    return build_queue(frame, version_id, threshold, rows=limit)


@router.get("/dataset-versions/{version_id}/review-queue")
def read_queue(
    version_id: str,
    request: Request,
    threshold: float | None = Query(default=None, ge=0, le=1),
    limit: int = Query(default=200, ge=1, le=1000),
) -> dict[str, Any]:
    """The labels below the threshold, with their answers and confidences.

    The configured threshold is used unless the caller overrides it, so the UI
    can offer a slider and re-read the queue without saving anything first.
    """
    from .projects import _require_version

    _require_version(request, version_id)
    effective = _threshold(request, threshold)
    queue = _queue_for(request, version_id, effective, limit)
    return {
        **queue.to_dict(),
        "decisions": list(DECISIONS),
    }


@router.get("/dataset-versions/{version_id}/review-status")
def review_status(
    version_id: str,
    request: Request,
    threshold: float | None = Query(default=None, ge=0, le=1),
) -> dict[str, Any]:
    """Just the outstanding count — what the Train step reads."""
    from .projects import _require_version

    _require_version(request, version_id)
    effective = _threshold(request, threshold)
    queue = _queue_for(request, version_id, effective, None)
    return {
        "version_id": version_id,
        "threshold": effective,
        "queued_count": queue.outstanding,
        "unlabeled_count": len(queue.unlabeled),
        "unreviewed_count": queue.unreviewed_count,
    }


@router.post("/dataset-versions/{version_id}/review", status_code=201)
def apply_review_decisions(
    version_id: str, body: ReviewIn, request: Request
) -> dict[str, Any]:
    """Apply accept / override / exclude decisions, producing a new version.

    An empty decision list is allowed: it produces a version with nothing
    changed, which is how the UI asks "what would still be unreviewed on this
    version?" without committing to anything.
    """
    from .projects import _require_version

    _require_version(request, version_id)
    effective = _threshold(request, body.threshold)
    try:
        decisions = validate_decisions([d.model_dump() for d in body.decisions])
    except ValueError as exc:
        raise HTTPException(422, str(exc)) from exc
    try:
        result = apply_review(request.app.state.store, version_id, decisions, threshold=effective)
    except ValueError as exc:
        raise HTTPException(422, str(exc)) from exc

    checks = []
    if body.raise_check:
        frame = request.app.state.store.load_dataframe(version_id, include_provenance=True)
        check = register_unreviewed_check(
            request.app.state.checks,
            version_id,
            unreviewed=result["unreviewed_count"],
            threshold=effective,
            total=len(label_families(frame)) * len(frame),
        )
        if check is not None:
            checks.append(check.to_dict())
    return {**result, "checks_raised": checks, "check_kind": UNREVIEWED_CHECK_KIND}


__all__ = ["DECISIONS", "router"]
