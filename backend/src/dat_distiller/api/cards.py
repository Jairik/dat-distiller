"""Card endpoints: the Markdown and JSON of a Dataset or Model Card.

Cards are the exportable record of how something was made, so they are served
in both forms: Markdown to read, JSON to read again with a tool. The same
content and the same redaction either way.
"""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, HTTPException, Query, Request, Response

from ..cards import dataset_card_for, model_card_for
from ..store.store import DatasetVersionNotFoundError

router = APIRouter(tags=["cards"])


def _checks_for(request: Request, subject_type: str, subject_id: str) -> list[dict[str, Any]]:
    return [check.to_dict() for check in request.app.state.checks.list_for_subject(subject_type, subject_id)]


@router.get("/dataset-versions/{version_id}/card")
def dataset_card(
    version_id: str,
    request: Request,
    format: str = Query("json", pattern="^(json|markdown)$"),
) -> Response:
    """The Dataset Card for one Dataset Version."""
    store = request.app.state.store
    try:
        store.get_version(version_id)
    except DatasetVersionNotFoundError as exc:
        raise HTTPException(404, "dataset version not found") from exc

    # The PII section reports what was *done* (from the version's own meta) and,
    # if nothing was done, what a fresh scan would find now. A Card must build
    # even when a scan cannot run, so a failure here degrades to the stored record.
    pii: dict[str, Any] = {}
    stored_pii = (store.get_version(version_id).meta or {}).get("pii") or {}
    if stored_pii:
        pii = dict(stored_pii)
    else:
        try:
            from ..pii import scan, summarize_findings

            pii = summarize_findings(scan(store.load_dataframe(version_id)))
        except Exception:  # noqa: BLE001 - a Card is not worth failing over
            pii = {"note": "no PII scan was available for this Dataset Version"}

    card = dataset_card_for(
        store,
        version_id,
        checks=_checks_for(request, "dataset_version", version_id),
        pii=pii,
    )
    if format == "markdown":
        return Response(
            content=card.to_markdown(),
            media_type="text/markdown; charset=utf-8",
            headers={"Content-Disposition": f'attachment; filename="dataset-card-v{card.number}.md"'},
        )
    return Response(
        content=_json(card.to_dict()),
        media_type="application/json",
        headers={"Content-Disposition": f'attachment; filename="dataset-card-v{card.number}.json"'},
    )


@router.get("/train/runs/{run_id}/card")
def model_card(
    run_id: str,
    request: Request,
    format: str = Query("json", pattern="^(json|markdown)$"),
) -> Response:
    """The Model Card for one Training Run, referencing its Dataset Card."""
    from ..jobs import JobNotFoundError
    from .train import _run_payload

    try:
        job = request.app.state.jobs.get(run_id)
    except JobNotFoundError as exc:
        raise HTTPException(404, f"no Training Run with id {run_id!r}") from exc
    run = _run_payload(job)

    dataset_summary = None
    version_id = str(run.get("version_id") or "")
    if version_id:
        try:
            summary = dataset_card_for(
                request.app.state.store,
                version_id,
                checks=_checks_for(request, "dataset_version", version_id),
            ).to_dict()
            dataset_summary = {
                "version_id": summary["version_id"],
                "number": summary["number"],
                "row_count": summary["row_count"],
                "origin": summary["origin"],
            }
        except DatasetVersionNotFoundError:
            dataset_summary = None

    card = model_card_for(
        run,
        checks=_checks_for(request, "training_run", run_id),
        dataset_card=dataset_summary,
    )
    if format == "markdown":
        return Response(
            content=card.to_markdown(),
            media_type="text/markdown; charset=utf-8",
            headers={"Content-Disposition": f'attachment; filename="model-card-{run_id[:8]}.md"'},
        )
    return Response(
        content=_json(card.to_dict()),
        media_type="application/json",
        headers={"Content-Disposition": f'attachment; filename="model-card-{run_id[:8]}.json"'},
    )


def _json(payload: dict[str, Any]) -> str:
    import json

    return json.dumps(payload, indent=2, default=str)


__all__ = ["model_card_for", "router"]
