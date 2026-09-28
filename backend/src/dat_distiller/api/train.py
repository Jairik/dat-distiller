"""Training setup endpoints: estimate, validate, the run job, and the plan.

The heavy fitting lives in :mod:`dat_distiller.training`; this router only
validates the request, hands it to the ``train_setup`` job, and reads the
checkpointed Training Run back.
"""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, HTTPException, Request

from ..jobs import JobNotFoundError
from ..store.store import DatasetVersionNotFoundError
from ..training.setup import (
    DEFAULT_REVIEW_THRESHOLD,
    TrainingSetupRequest,
    build_setup,
    estimate_setup,
)
from ..training.task import infer_task_type, sibling_columns

router = APIRouter(tags=["train"])

JOB_TYPE = "train_setup"


def _load(request: Request, body: TrainingSetupRequest) -> tuple[Any, Any, Any]:
    """The store, the Dataset Version record and its frame (Provenance included)."""
    store = request.app.state.store
    try:
        version = store.get_version(body.version_id)
    except DatasetVersionNotFoundError as exc:
        raise HTTPException(404, "dataset version not found") from exc
    return store, version, store.load_dataframe(body.version_id, include_provenance=True)


def _threshold(request: Request) -> float:
    return float(
        request.app.state.settings.load().get("review_threshold", DEFAULT_REVIEW_THRESHOLD)
    )


def _resolve(body: TrainingSetupRequest, frame: Any, request: Request):
    try:
        return build_setup(body, frame, review_threshold=_threshold(request))
    except ValueError as exc:
        raise HTTPException(422, str(exc)) from exc


@router.post("/train/estimate")
def estimate(body: TrainingSetupRequest, request: Request) -> dict[str, Any]:
    """What the Training Run would do — rows, features, split and seed."""
    _, _, frame = _load(request, body)
    try:
        return estimate_setup(body, frame, review_threshold=_threshold(request))
    except ValueError as exc:
        raise HTTPException(422, str(exc)) from exc


@router.post("/train/validate")
def validate(body: TrainingSetupRequest, request: Request) -> dict[str, Any]:
    """Check the request against a Dataset Version and describe the plan."""
    _, _, frame = _load(request, body)
    inferred = infer_task_type(frame[body.target]) if body.target in frame.columns else None
    setup = _resolve(body, frame, request).setup
    return {
        "valid": True,
        "target": setup.target,
        "task_type": setup.task_type,
        "inferred_task_type": inferred,
        "task_type_changed": inferred is not None and inferred != setup.task_type,
        "label_family": setup.label_family,
        "siblings": sorted(sibling_columns(body.target, frame.columns)),
        "feature_columns": list(setup.feature_columns),
        "excluded_columns": dict(setup.excluded_columns),
        "n_source_features": len(setup.feature_columns),
        "sensitive_attribute": setup.sensitive_attribute,
        "sensitive_attribute_included": setup.include_sensitive_attribute,
        "exclude_unreviewed": setup.exclude_unreviewed,
        "review_threshold": setup.review_threshold,
        "confidence_column": setup.confidence_column,
        "rows": len(frame),
        "kept_rows": setup.kept_rows,
        "train_rows": len(setup.split.train),
        "test_rows": len(setup.split.test),
        "test_size": setup.test_size,
        "stratified": setup.split.stratified,
        "seed": setup.seed,
        "class_counts": dict(setup.class_counts),
        "unreviewed_rows": setup.unreviewed_rows,
        "dropped_unreviewed": setup.dropped_unreviewed,
    }


@router.post("/train/setup", status_code=202)
def run(body: TrainingSetupRequest, request: Request) -> dict[str, Any]:
    """Start the Training Run's setup as a background job.

    The result carries the fitted preprocessing pipeline, the split and the
    Checks; the leaderboard (#32) and the Model Bundle (#34) read them from the
    job's checkpoint.
    """
    _, version, frame = _load(request, body)
    _resolve(body, frame, request)  # fail fast on an impossible plan
    try:
        job = request.app.state.jobs.start(
            JOB_TYPE, project_id=version.project_id, params=body.to_params()
        )
    except ValueError as exc:
        raise HTTPException(422, str(exc)) from exc
    return job.to_dict()


@router.get("/train/setup/{run_id}")
def get_setup(run_id: str, request: Request) -> dict[str, Any]:
    """The Training Run's checkpointed plan: features, split, pipeline, Checks."""
    jobs = request.app.state.jobs
    try:
        job = jobs.get(run_id)
    except JobNotFoundError as exc:
        raise HTTPException(404, "training run not found") from exc
    if job.type != JOB_TYPE:
        raise HTTPException(404, "not a training setup run")
    checks: Any = request.app.state.checks
    checkpoint = job.checkpoint or {}
    return {
        "training_run_id": run_id,
        "job_id": job.id,
        "project_id": job.project_id,
        "status": job.status,
        "setup": checkpoint.get("setup", {}),
        "preprocessing": checkpoint.get("preprocessing", {}),
        "split_indices": {
            "train": checkpoint.get("train_indices", []),
            "test": checkpoint.get("test_indices", []),
        },
        "checks": [check.to_dict() for check in checks.list_for_subject("training_run", run_id)],
        "unacknowledged_warnings": [
            check.to_dict() for check in checks.unacknowledged_warnings("training_run", run_id)
        ],
        "result": job.result,
    }
