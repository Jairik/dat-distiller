"""PII API: scan a Dataset Version, and apply the mask / drop actions.

The ``warn`` action has no endpoint of its own on purpose — the Acknowledgement
of the ``pii_found`` Check *is* the warn path, and it already exists. Only the
two destructive actions get an endpoint, because both create a new Dataset
Version rather than mutating the one you looked at.
"""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel, Field

from ..pii import (
    ACTIONS,
    PII_CHECK_KIND,
    apply_actions,
    register_pii_checks,
    scan,
    summarize_findings,
)
from .projects import _require_version

router = APIRouter(tags=["pii"])


class PiiActionsIn(BaseModel):
    """`{column: "mask" | "drop"}` for the columns the user chose to act on."""

    actions: dict[str, str] = Field(default_factory=dict)

    def validated(self) -> dict[str, str]:
        bad = {c: a for c, a in self.actions.items() if a not in ACTIONS}
        if bad:
            raise ValueError(f"pii action must be one of {ACTIONS}, got {sorted(set(bad.values()))}")
        return dict(self.actions)


@router.get("/dataset-versions/{version_id}/pii")
def scan_version(version_id: str, request: Request) -> dict[str, Any]:
    """Findings for one Dataset Version, without registering any Check.

    Read-only: this is the "what would you do about it?" call the UI makes
    before the user picks warn / mask / drop. Checks are raised by the upload and
    Generation hooks, or by `POST .../pii/warn` below.
    """
    version = _require_version(request, version_id)
    store = request.app.state.store
    df = store.load_dataframe(version.id, include_provenance=False)
    findings = scan(df)
    return {
        "version_id": version.id,
        "findings": findings,
        "summary": summarize_findings(findings),
        "actions": list(ACTIONS),
    }


@router.post("/dataset-versions/{version_id}/pii/warn")
def warn_version(version_id: str, request: Request) -> dict[str, Any]:
    """The `warn` action: raise a warning Check per affected column, unchanged data.

    Nothing is blocked — acknowledging each Check is the whole of the contract,
    and ChecksPanel (#15) is what waits on that.
    """
    version = _require_version(request, version_id)
    store = request.app.state.store
    df = store.load_dataframe(version.id, include_provenance=False)
    checks = register_pii_checks(request.app.state.checks, version.id, scan(df))
    return {
        "version_id": version.id,
        "checks": [check.to_dict() for check in checks],
        "check_kind": PII_CHECK_KIND,
    }


@router.post("/dataset-versions/{version_id}/pii/actions", status_code=201)
def apply_pii_actions(version_id: str, body: PiiActionsIn, request: Request) -> dict[str, Any]:
    """The `mask` / `drop` actions: a new Dataset Version, parented to this one.

    The source Version is immutable, so the parent keeps the PII and the child
    records what was done to it in `meta` — that record is what the Dataset Card
    reads. The actions are recorded even when they turned out to be no-ops, so
    the Card can say "masked, found nothing" rather than implying silence.
    """
    try:
        actions = body.validated()
    except ValueError as exc:
        raise HTTPException(422, str(exc)) from exc
    version = _require_version(request, version_id)
    store = request.app.state.store
    df = store.load_dataframe(version.id, include_provenance=True)
    try:
        cleaned, summary = apply_actions(df, actions)
    except ValueError as exc:
        raise HTTPException(422, str(exc)) from exc

    child = store.create_version(
        version.project_id,
        cleaned,
        parent_id=version.id,
        origin="redacted",
        meta={
            "pii": {
                "parent_version_id": version.id,
                "actions": actions,
                "summary": summary,
            }
        },
    )
    # the child was scanned by a different rule (it no longer holds what the
    # parent did), so findings are reported for information, not raised as Checks
    remaining = scan(store.load_dataframe(child.id, include_provenance=False))
    return {
        "version": child.to_dict(),
        "summary": summary,
        "remaining_findings": summarize_findings(remaining),
    }


__all__ = ["ACTIONS", "PII_CHECK_KIND", "router"]
