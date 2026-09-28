"""The Fairness Report endpoint: computed on demand, never at training time.

``POST /api/train/runs/{run_id}/fairness`` is the only way a Fairness Report comes
into existence. Nothing in the Training Run (#32) computes one, so a run that was
never asked stays exactly as it was — no extra fitting, no extra Checks.

The endpoint reads its threshold from ``settings.fairness_gap_threshold`` unless
the caller names one, exactly as the Review Queue reads ``review_threshold``, so
the UI can offer a slider and re-measure without saving anything first.
"""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel, Field

from ..jobs import TERMINAL_STATUSES, JobNotFoundError
from ..training.fairness import (
    DEFAULT_GAP_THRESHOLD,
    FAIRNESS_STEP,
    FairnessError,
    fairness_report,
    register_fairness_checks,
)
from .train import RUN_JOB_TYPE

router = APIRouter(tags=["fairness"])


class FairnessRequestIn(BaseModel):
    """Which Model, split by which attribute, judged against which threshold."""

    #: The column whose groups are compared. Excluded from the features by
    #: default since #31; the report says so either way.
    sensitive_attribute: str = Field(min_length=1)
    #: ``None`` means the top-ranked Model on this run's leaderboard.
    model: str | None = None
    #: Which class counts as positive for TPR/FPR/selection rate. ``None`` means
    #: the last class in the Target's sorted order.
    positive_label: str | None = None
    #: Overrides ``settings.fairness_gap_threshold`` for this request only.
    gap_threshold: float | None = Field(default=None, ge=0.0, le=1.0)
    #: Set ``false`` to measure without raising the report's Checks.
    raise_check: bool = True


def _configured_threshold(request: Request) -> tuple[float, str]:
    """``settings.fairness_gap_threshold``, or the module default when unset."""
    raw = request.app.state.settings.load().get("fairness_gap_threshold")
    if raw is None:
        return DEFAULT_GAP_THRESHOLD, "default"
    return float(raw), "settings"


@router.post("/train/runs/{run_id}/fairness")
def report(
    run_id: str, body: FairnessRequestIn, request: Request
) -> dict[str, Any]:
    """The Fairness Report for one Model of a Training Run, by one Sensitive Attribute.

    Reuses the run's own split, preprocessing and hyperparameters, refits the
    Model from them, and predicts the held-out rows once. Per-group accuracy, TPR,
    FPR and the demographic parity difference for a classification Target;
    per-group MAE for a regression Target. Any gap wider than the threshold
    becomes a warning **Check** on the Training Run.

    The response is the report plus ``checks_raised``. Gaps that clear the
    threshold are ordered widest first in ``gaps_exceeding_threshold``, and
    ``largest_gap`` names the widest gap overall so a reader never has to guess
    which gap is the finding.
    """
    jobs = request.app.state.jobs
    try:
        job = jobs.get(run_id)
    except JobNotFoundError as exc:
        raise HTTPException(404, "training run not found") from exc
    if job.type != RUN_JOB_TYPE:
        raise HTTPException(404, "not a training run")
    if job.status not in TERMINAL_STATUSES:
        raise HTTPException(
            409,
            f"this Training Run is {job.status}; a Fairness Report needs a finished run, because "
            "it re-measures the Model that run fitted",
        )

    configured, source = _configured_threshold(request)
    threshold = body.gap_threshold if body.gap_threshold is not None else configured
    threshold_source = "request" if body.gap_threshold is not None else source
    try:
        result = fairness_report(
            store=request.app.state.store,
            job=job,
            sensitive_attribute=body.sensitive_attribute,
            model=body.model,
            positive_label=body.positive_label,
            threshold=threshold,
            threshold_source=threshold_source,
        )
    except FairnessError as exc:
        raise HTTPException(422, str(exc)) from exc

    checks = request.app.state.checks
    raised = (
        [check.to_dict() for check in register_fairness_checks(checks, result, step=FAIRNESS_STEP)]
        if body.raise_check
        else []
    )
    return {
        **result,
        "checks_raised": raised,
        "check_step": FAIRNESS_STEP,
        "unacknowledged_warnings": [
            check.to_dict()
            for check in checks.unacknowledged_warnings("training_run", run_id)
        ],
    }


__all__ = ["router"]
