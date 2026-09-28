"""On-demand evaluation and prediction endpoints.

Nothing here is wired into a Training Run. A Training Run finishes; these are
called afterwards by whoever wants a plot or a prediction, and each call pays its
own cost. A run that never calls them does no evaluation work at all.

Prediction results are kept in a small bounded in-memory cache so the download
URL is a real URL rather than a promise. Single user, single machine, and the
contents are derived from data the user already has.
"""

from __future__ import annotations

import json
import secrets
from collections import OrderedDict
from typing import Any

from fastapi import APIRouter, HTTPException, Query, Request, Response, UploadFile

from ..ingest import (
    MalformedFileError,
    UnsupportedFormatError,
    read_dataframe,
    to_csv_bytes,
)
from ..store.store import DatasetVersionNotFoundError
from ..training.evaluate import (
    PLOT_TASK_TYPES,
    PLOTS,
    EvaluationError,
    NotOnRunError,
    evaluate_run,
    predict_frame,
    refit,
)
from ..training.run_rows import RunContext, resolve_run

router = APIRouter(tags=["train"])

#: How many recent prediction files to keep for download. Small on purpose:
#: these are derivable from the Model and the uploaded file, not records.
PREDICTION_CACHE_SIZE = 16

_predictions: OrderedDict[str, tuple[bytes, str]] = OrderedDict()


def _remember_predictions(body: bytes, filename: str) -> str:
    token = secrets.token_urlsafe(12)
    _predictions[token] = (body, filename)
    while len(_predictions) > PREDICTION_CACHE_SIZE:
        _predictions.popitem(last=False)
    return token


def _job(request: Request, run_id: str) -> Any:
    """The Training Run's job, or a 404 that says so plainly."""
    from ..jobs import JobNotFoundError

    try:
        return request.app.state.jobs.get(run_id)
    except JobNotFoundError as exc:
        raise HTTPException(404, f"no Training Run with id {run_id!r}") from exc


def _run(request: Request, run_id: str) -> dict[str, Any]:
    """The persisted Training Run, or a 404 that says so plainly."""
    from .train import _run_payload  # local import avoids a cycle at module load

    return _run_payload(_job(request, run_id))


def _run_rows(request: Request, run_id: str) -> RunContext:
    """The rows the run actually worked on, rebuilt and checked.

    A Training Run does not work on the Dataset Version as stored: it drops rows
    with no Target value and rows whose labelling is still below the Review
    threshold, and its split indices address what is left over. Reading the
    stored version instead aims every index at the wrong row — silently, because
    the indices are all still in range. So this goes through the same resolver a
    Fairness Report uses, which rebuilds the filtered frame and refuses if the
    stored split no longer reproduces.
    """
    try:
        return resolve_run(request.app.state.store, _job(request, run_id))
    except DatasetVersionNotFoundError as exc:
        raise HTTPException(404, "the Dataset Version this run used is gone") from exc
    except ValueError as exc:
        raise HTTPException(422, str(exc)) from exc


@router.get("/train/runs/{run_id}/plots")
def run_plot(
    run_id: str,
    request: Request,
    model: str = Query(..., min_length=1),
    plot: str = Query(..., min_length=1),
    repeats: int = Query(5, ge=1, le=50),
) -> dict[str, Any]:
    """One plot for one Model, computed now.

    A plot the Model cannot produce (ROC for a Model with no probabilities) comes
    back as 200 with an `error` explaining why, rather than a 500: the UI is
    drawing a row of cards, and one impossible card should not take out the rest.
    """
    run = _run(request, run_id)
    if plot not in PLOTS:
        raise HTTPException(422, f"plot must be one of {PLOTS}, got {plot!r}")
    expected = PLOT_TASK_TYPES[plot]
    task_type = str(run.get("task_type") or "")
    if "+" not in expected and expected != task_type:
        raise HTTPException(
            422, f"the {plot} plot needs a {expected} Target; this run's is {task_type}"
        )
    context = _run_rows(request, run_id)
    try:
        return evaluate_run(run, model, context.data_frame, plot=plot, repeats=repeats)
    except NotOnRunError as exc:
        # asking for a Model that was not fitted is a mistake in the request
        raise HTTPException(422, str(exc)) from exc
    except EvaluationError as exc:
        # the Model exists but cannot produce this plot: answered, not refused,
        # so one impossible card does not take out the whole panel
        return {"plot": plot, "model": model, "error": str(exc)}


@router.post("/train/runs/{run_id}/predict")
async def run_predict(
    run_id: str,
    request: Request,
    file: UploadFile,
    model: str = Query(..., min_length=1),
) -> dict[str, Any]:
    """Predict with a chosen Model on a new CSV.

    The response is the input plus `prediction` (and a probability per class for
    a classification Model), plus a `download_url` for the same thing as a file.
    """
    run = _run(request, run_id)
    context = _run_rows(request, run_id)
    content = await file.read()
    if not content:
        raise HTTPException(422, "uploaded file is empty")
    try:
        frame = read_dataframe(file.filename or "predict.csv", content)
    except UnsupportedFormatError as exc:
        raise HTTPException(415, str(exc)) from exc
    except MalformedFileError as exc:
        raise HTTPException(422, str(exc)) from exc

    try:
        predicted = predict_frame(run, model, frame, refit(run, model, context.data_frame))
    except EvaluationError as exc:
        raise HTTPException(422, str(exc)) from exc

    columns = [str(c) for c in predicted.columns]
    rows = json.loads(predicted.to_json(orient="records", date_format="iso"))
    token = _remember_predictions(to_csv_bytes(predicted), f"predictions-{run_id[:8]}.csv")
    return {
        "run_id": run_id,
        "model": model,
        "rows": len(rows),
        "columns": columns,
        "predictions": rows,
        "download_url": f"/api/train/runs/{run_id}/predictions.csv?model={model}&token={token}",
    }


@router.get("/train/runs/{run_id}/predictions.csv")
def download_predictions(run_id: str, request: Request, token: str = Query(...)) -> Response:
    """Serve a prediction file produced by `run_predict`."""
    entry = _predictions.get(token)
    if entry is None:
        raise HTTPException(
            404, "that prediction file is no longer available — predict again to get a new one"
        )
    body, filename = entry
    return Response(
        content=body,
        media_type="text/csv",
        headers={"Content-Disposition": f'attachment; filename="{filename}"'},
    )


__all__ = ["PLOTS", "PREDICTION_CACHE_SIZE", "router"]
