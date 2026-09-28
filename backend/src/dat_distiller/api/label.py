"""Labeling endpoints: preview, estimate, and the resumable run job."""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, HTTPException, Request

from ..jev import JevError, JevNotConfiguredError, resolve_jev
from ..label import (
    LabelRequest,
    estimate_labeling,
    label_columns,
    parse_question,
    preview_labeling,
    state_columns_for,
)
from ..store.store import DatasetVersionNotFoundError

router = APIRouter(tags=["label"])


def _load(request: Request, version_id: str):
    store = request.app.state.store
    try:
        return store, store.load_dataframe(version_id)
    except DatasetVersionNotFoundError as exc:
        raise HTTPException(404, "dataset version not found") from exc


def _questions_or_422(specs):
    try:
        return [parse_question(spec) for spec in specs]
    except ValueError as exc:
        raise HTTPException(422, str(exc)) from exc


@router.post("/label/preview")
def preview(body: LabelRequest, request: Request) -> dict[str, Any]:
    """Run the Jev Questions over a few rows before committing to a Labeling run."""
    store, frame = _load(request, body.version_id)
    questions = _questions_or_422(body.questions)
    outputs = label_columns(questions)
    try:
        state_columns = state_columns_for(frame, body.state_columns, outputs)
    except ValueError as exc:
        raise HTTPException(422, str(exc)) from exc
    try:
        jev = resolve_jev(request.app.state.settings, model=body.jev_model)
        rows = preview_labeling(jev, frame, questions, state_columns, body.preview_rows)
    except JevNotConfiguredError as exc:
        raise HTTPException(422, str(exc)) from exc
    except JevError as exc:
        raise HTTPException(502, f"Jev call failed: {exc}") from exc
    return {"state_columns": state_columns, "rows": rows}


@router.post("/label/estimate")
def estimate(body: LabelRequest, request: Request) -> dict[str, Any]:
    _, frame = _load(request, body.version_id)
    _questions_or_422(body.questions)
    return estimate_labeling(len(frame))


@router.post("/label/run", status_code=202)
def run(body: LabelRequest, request: Request) -> dict[str, Any]:
    """Start the Labeling run as a resumable background job."""
    store, frame = _load(request, body.version_id)
    _questions_or_422(body.questions)
    version = store.get_version(body.version_id)
    job = request.app.state.jobs.start(
        "label", project_id=version.project_id, params=body.model_dump()
    )
    return job.to_dict()
