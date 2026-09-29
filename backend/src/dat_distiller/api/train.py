"""Training endpoints: the setup plan, the Training Run, and the leaderboard.

Two steps, two jobs:

- ``train_setup`` (#31) resolves the Target, the features, the leakage guard, the
  preprocessing and the held-out split, and checkpoints them.
- ``train`` (#32) fits the selected **Models** — tuning them with
  cross-validation on the **training split only** — and ranks them on the
  held-out test split. Its job id *is* the Training Run id.

The fitting lives in :mod:`dat_distiller.training`; this router only validates
the request, hands it to the job, and reads the persisted Training Run back.
"""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, HTTPException, Query, Request

from ..extras import installed_extras
from ..jobs import JobNotFoundError
from ..store.store import DatasetVersionNotFoundError
from ..training import trainers
from ..training.setup import (
    DEFAULT_REVIEW_THRESHOLD,
    TrainingSetupRequest,
    build_setup,
    estimate_setup,
)
from ..training.task import infer_task_type, sibling_columns
from ..training.train import TrainingRunRequest, plan_training_run

router = APIRouter(tags=["train"])

JOB_TYPE = "train_setup"
RUN_JOB_TYPE = "train"


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
def run_setup(body: TrainingSetupRequest, request: Request) -> dict[str, Any]:
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


# -- the Model registry ------------------------------------------------------


@router.get("/train/models")
def models() -> dict[str, Any]:
    """Every Model: the Task Types it supports, its defaults, its search space,
    and whether this install can run it. The UI builds its picker from this."""
    return {
        "models": [
            {
                **spec.to_dict(),
                "available": trainers.is_available(spec),
                "class_path": {
                    task_type: spec.path(task_type) for task_type in spec.task_types
                },
            }
            for spec in trainers.MODEL_SPECS.values()
        ],
        "extras": installed_extras(),
        "library_versions": trainers.library_versions(),
    }


@router.get("/train/metrics")
def metrics(task_type: str | None = Query(default=None)) -> dict[str, Any]:
    """The metrics a Task Type can produce, and its default primary metric.

    ``task_type`` narrows the answer; without it both are returned, so the UI can
    offer the right dropdown the moment the Task Type is inferred.
    """
    task_types = [task_type] if task_type else list(trainers.TASK_TYPES)
    try:
        return {
            "task_types": {
                name: {
                    "metrics": [
                        trainers.METRICS[metric].to_dict()
                        for metric in trainers.metrics_for(name)
                    ],
                    "default_primary_metric": trainers.default_primary_metric(name),
                }
                for name in task_types
            }
        }
    except trainers.ModelError as exc:
        raise HTTPException(422, str(exc)) from exc


# -- planning the Training Run ----------------------------------------------


@router.post("/train/plan")
def plan(body: TrainingRunRequest, request: Request) -> dict[str, Any]:
    """What the Training Run would train and rank, without training anything.

    Refuses a Model that does not support the Target's Task Type, a Model whose
    extra is not installed, and a primary metric the Task Type cannot produce.
    """
    _, _, frame = _load(request, body)
    try:
        return plan_training_run(body, frame, review_threshold=_threshold(request))
    except ValueError as exc:
        raise HTTPException(422, str(exc)) from exc


@router.post("/train/run", status_code=202)
def run(body: TrainingRunRequest, request: Request) -> dict[str, Any]:
    """Start a Training Run: fit the selected Models and rank them.

    Tuning, when asked for, cross-validates on the **training split only**; the
    held-out test split is touched once, to score the finished Models.
    """
    _, version, frame = _load(request, body)
    try:
        plan_training_run(body, frame, review_threshold=_threshold(request))
    except ValueError as exc:
        raise HTTPException(422, str(exc)) from exc
    try:
        job = request.app.state.jobs.start(
            RUN_JOB_TYPE, project_id=version.project_id, params=body.to_params()
        )
    except ValueError as exc:
        raise HTTPException(422, str(exc)) from exc
    return job.to_dict()


# -- reading a Training Run back -------------------------------------------


def _run_payload(job: Any) -> dict[str, Any]:
    """The persisted Training Run, whether it finished or is still running."""
    checkpoint = job.checkpoint or {}
    run = checkpoint.get("run")
    if run is not None:
        return dict(run)
    if job.result is not None:
        return dict(job.result)
    return {
        "training_run_id": job.id,
        "project_id": job.project_id,
        "status": job.status,
        "setup": checkpoint.get("setup", {}),
        "preprocessing": checkpoint.get("preprocessing", {}),
        "hyperparameters": checkpoint.get("hyperparameters", {}),
        "tuning": checkpoint.get("tuning", {}),
        "training_split": checkpoint.get("training_split", {}),
        "test_split": {
            "rows": len(checkpoint.get("test_indices") or []),
            "indices": checkpoint.get("test_indices", []),
        },
        "leaderboard": checkpoint.get("leaderboard", []),
        "warnings": [],
        "finished": False,
    }


@router.get("/train/runs")
def list_runs(
    request: Request,
    project_id: str = Query(min_length=1),
    limit: int = Query(default=50, ge=1, le=200),
) -> dict[str, Any]:
    """The Training Runs of a Project, newest first, with their leaderboard heads."""
    jobs = [
        job
        for job in request.app.state.jobs.list_for_project(project_id)
        if job.type == RUN_JOB_TYPE
    ]
    runs = [_summary(job) for job in reversed(jobs[-limit:])]
    return {"project_id": project_id, "runs": runs, "count": len(runs)}


def _summary(job: Any) -> dict[str, Any]:
    run = _run_payload(job)
    leaderboard = run.get("leaderboard") or []
    ranked = [entry for entry in leaderboard if entry.get("rank") is not None]
    return {
        "training_run_id": job.id,
        "job_id": job.id,
        "project_id": job.project_id,
        "status": job.status,
        "progress": job.progress,
        "created_at": job.created_at,
        "finished_at": job.finished_at,
        "version_id": run.get("version_id"),
        "target": run.get("target"),
        "task_type": run.get("task_type"),
        "seed": run.get("seed"),
        "primary_metric": run.get("primary_metric"),
        # Without this the list of past runs has to *guess* the direction and
        # defaults to "higher is better", so a run whose primary metric is rmse
        # or mae is summarised with an up arrow next to the worse number.
        "primary_metric_higher_is_better": run.get("primary_metric_higher_is_better"),
        "n_models": len(leaderboard),
        "best_model": ranked[0]["model"] if ranked else None,
        "best_primary_value": (ranked[0].get("primary") or {}).get("value") if ranked else None,
        "warnings": list(run.get("warnings") or []),
        "error": job.error,
    }


@router.get("/train/runs/{run_id}")
def get_run(run_id: str, request: Request) -> dict[str, Any]:
    """The persisted Training Run: seeds, library versions, hyperparameters,
    the tuning search, and the leaderboard of held-out metrics."""
    try:
        job = request.app.state.jobs.get(run_id)
    except JobNotFoundError as exc:
        raise HTTPException(404, "training run not found") from exc
    if job.type != RUN_JOB_TYPE:
        raise HTTPException(404, "not a training run")
    checks: Any = request.app.state.checks
    run = _run_payload(job)
    return {
        **run,
        "job_id": job.id,
        "status": job.status,
        "progress": job.progress,
        "created_at": job.created_at,
        "finished_at": job.finished_at,
        "error": job.error,
        "setup_run_id": (job.params or {}).get("setup_run_id"),
        "split_indices": {
            "train": (job.checkpoint or {}).get("train_indices", []),
            "test": (job.checkpoint or {}).get("test_indices", []),
        },
        "checks": [check.to_dict() for check in checks.list_for_subject("training_run", run_id)],
        "unacknowledged_warnings": [
            check.to_dict() for check in checks.unacknowledged_warnings("training_run", run_id)
        ],
    }
