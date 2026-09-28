"""The Model Bundle download endpoint.

One route, one thing: put a Model in a zip that someone else can load. The
Model Card goes *inside* the bundle, because a bundle that separates a Model from
its caveats invites someone to share the first and not the second.
"""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, HTTPException, Query, Request, Response

from ..jobs import JobNotFoundError
from ..store.store import DatasetVersionNotFoundError
from ..training.bundle import BundleError, build_bundle
from ..training.evaluate import NotOnRunError

router = APIRouter(tags=["train"])


def _run(request: Request, run_id: str) -> dict[str, Any]:
    from .train import _run_payload

    try:
        job = request.app.state.jobs.get(run_id)
    except JobNotFoundError as exc:
        raise HTTPException(404, f"no Training Run with id {run_id!r}") from exc
    return _run_payload(job)


def _frame(request: Request, run: dict[str, Any]):
    version_id = str(run.get("version_id") or "")
    if not version_id:
        raise HTTPException(422, "this Training Run does not record its Dataset Version")
    try:
        return request.app.state.store.load_dataframe(version_id)
    except DatasetVersionNotFoundError as exc:
        raise HTTPException(404, "the Dataset Version this run used is gone") from exc


def _model_card(request: Request, run: dict[str, Any]) -> str | None:
    """The Model Card as Markdown, or None if it cannot be built.

    A bundle without its Card is a real gap, so it is reported in the download
    rather than quietly shipped.
    """
    try:
        from ..cards import model_card_for

        run_id = str(run.get("training_run_id") or "")
        checks = [
            check.to_dict()
            for check in request.app.state.checks.list_for_subject("training_run", run_id)
        ]
        card = model_card_for(run, checks=checks)
        return card.to_markdown()
    except Exception:  # noqa: BLE001 - a Card is never worth failing a download over
        return None


@router.get("/train/runs/{run_id}/bundle")
def download_bundle(
    run_id: str,
    request: Request,
    model: str = Query(..., min_length=1),
    onnx: bool = Query(True, description="Attempt ONNX conversion; failure is not fatal."),
) -> Response:
    """The fitted Model, its pipeline, its metadata and its Card, as a zip."""
    run = _run(request, run_id)
    try:
        payload = build_bundle(
            run,
            model,
            _frame(request, run),
            model_card=_model_card(request, run),
            with_onnx=onnx,
        )
    except NotOnRunError as exc:
        raise HTTPException(422, str(exc)) from exc
    except BundleError as exc:
        raise HTTPException(422, str(exc)) from exc
    return Response(
        content=payload,
        media_type="application/zip",
        headers={
            "Content-Disposition": f'attachment; filename="{model}-bundle.zip"',
            "Content-Length": str(len(payload)),
        },
    )


__all__ = ["router"]
