"""The Training Run job: fit the selected Models, tune them, keep a leaderboard.

One run of this module is one **Training Run** — the job id *is* the Training
Run id, and everything about it lands in the job's checkpoint: the resolved plan
and preprocessing from #31, the seeds, the library versions, the
hyperparameters each Model was fitted with, the tuning search that produced
them, and the leaderboard of metrics measured on the held-out test split.

The split discipline is the point of the module and it is structural, not a
convention:

1. :meth:`TrainingMatrix.from_split` is the only way to build the matrix, and it
   reads ``setup.split.train`` only. :func:`trainers.tune_model` accepts nothing
   but a ``TrainingMatrix``, so the search *cannot* be handed the test rows.
2. The test matrix is built **after** every Model has been fitted and tuned, in
   :func:`_score`, and is only ever used to score.
3. :func:`trainers.assert_held_out` refuses a training matrix that overlaps the
   held-out indices, so a future refactor that mixes the two fails loudly.
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from typing import Any, Literal

from pydantic import BaseModel, Field

from ..checks import CheckStore
from ..jobs import JobContext
from ..store import DatasetStore
from . import trainers
from .checks import evaluate_training_checks, raise_training_checks
from .preprocess import Preprocessor, fit_preprocessor, transform_with_spec
from .setup import (
    DEFAULT_REVIEW_THRESHOLD,
    TrainingSetupRequest,
    build_setup,
    effective_seed,
)
from .task import label_family
from .trainers import (
    METRICS,
    MODEL_SPECS,
    ModelError,
    ModelSpec,
    TrainingMatrix,
)

#: Stages the job reports progress over.
STAGES = ("setup", "matrix", "tune", "fit", "score", "checks", "done")


class TuningConfig(BaseModel):
    """Optional hyperparameter search, run on the training split only."""

    enabled: bool = False
    #: ``random`` samples ``n_iter`` candidates; ``grid`` tries every combination.
    strategy: Literal["random", "grid"] = "random"
    n_iter: int = Field(default=10, ge=1, le=500)
    cv: int = Field(default=3, ge=2, le=20)
    #: Per-Model override of the built-in search space, e.g.
    #: ``{"svm": {"C": [0.5, 1.0, 5.0]}}``.
    search_space: dict[str, dict[str, list[Any]]] = Field(default_factory=dict)
    #: Metric the search maximises/minimises. Defaults to the run's primary metric.
    metric: str | None = None


class TrainingRunRequest(TrainingSetupRequest):
    """The setup request plus which Models to train and how to rank them."""

    #: ``None`` means every Model this install can run for the Target's Task Type.
    models: list[str] | None = None
    #: ``None`` means the Task Type's default (F1 macro for classification, R²
    #: for regression).
    primary_metric: str | None = None
    tuning: TuningConfig = Field(default_factory=TuningConfig)
    #: Hyperparameters pinned per Model, e.g. ``{"svm": {"C": 5.0}}``.
    hyperparameters: dict[str, dict[str, Any]] = Field(default_factory=dict)
    #: Reuse the preprocessing a previous ``train_setup`` run checkpointed.
    setup_run_id: str | None = None

    def to_params(self) -> dict[str, Any]:
        return self.model_dump()


def split_request(request: TrainingRunRequest) -> TrainingSetupRequest:
    """The setup half of a Training Run request.

    The held-out split has to depend on the Target, the features, the leakage
    guard, ``test_size`` and an explicit seed — and on nothing else. Deriving the
    seed from the whole request would mean that ticking one more Model off the
    list silently re-splits the data, and two leaderboards could not be compared.
    """
    dumped = request.model_dump()
    return TrainingSetupRequest.model_validate(
        {name: dumped[name] for name in TrainingSetupRequest.model_fields}
    )


# -- planning ----------------------------------------------------------------


def _unavailable(names: Sequence[str] | None) -> list[dict[str, str]]:
    """Selected Models this install cannot run, with the extra to install."""
    if names is None:
        return [
            {"model": name, "reason": f"needs the optional extra {spec.extra!r}"}
            for name, spec in MODEL_SPECS.items()
            if not trainers.is_available(spec)
        ]
    out: list[dict[str, str]] = []
    for name in names:
        spec = MODEL_SPECS.get(str(name))
        if spec is not None and not trainers.is_available(spec):
            out.append({"model": name, "reason": f"needs the optional extra {spec.extra!r}"})
    return out


def _is_tunable(spec: ModelSpec, tuning: TuningConfig, task_type: str) -> bool:
    if not tuning.enabled:
        return False
    metric = trainers.resolve_primary_metric(tuning.metric, task_type)
    if METRICS[metric].needs_proba and not spec.supports_proba:
        return False
    return bool(tuning.search_space.get(spec.name)) or bool(spec.search_space)


def plan_training_run(
    request: TrainingRunRequest,
    frame: Any,
    *,
    review_threshold: float = DEFAULT_REVIEW_THRESHOLD,
) -> dict[str, Any]:
    """Resolve the plan before anything is fitted and refuse an impossible
    Model/Task Type pairing. Cheap enough for the UI to call on every change."""
    resolved = build_setup(split_request(request), frame, review_threshold=review_threshold)
    setup = resolved.setup
    specs = trainers.resolve_models(request.models, setup.task_type)
    primary_metric = trainers.resolve_primary_metric(request.primary_metric, setup.task_type)
    unknown = sorted(set(request.hyperparameters) - set(MODEL_SPECS))
    if unknown:
        raise ModelError(f"hyperparameters given for unknown Model(s): {', '.join(unknown)}")
    not_selected = sorted(set(request.hyperparameters) - {spec.name for spec in specs})
    if not_selected:
        raise ModelError(
            "hyperparameters given for Model(s) that are not being trained: "
            + ", ".join(not_selected)
        )
    seed = effective_seed(split_request(request))
    tuning_metric = trainers.resolve_primary_metric(request.tuning.metric, setup.task_type)
    return {
        "version_id": setup.version_id,
        "target": setup.target,
        "task_type": setup.task_type,
        "label_family": setup.label_family,
        "seed": seed,
        "rows": len(frame),
        "kept_rows": setup.kept_rows,
        "train_rows": len(setup.split.train),
        "test_rows": len(setup.split.test),
        "n_source_features": len(setup.feature_columns),
        "models": [spec.to_dict() for spec in specs],
        "model_names": [spec.name for spec in specs],
        "unavailable_models": _unavailable(request.models),
        "primary_metric": primary_metric,
        "primary_metric_label": METRICS[primary_metric].label,
        "primary_metric_higher_is_better": METRICS[primary_metric].higher_is_better,
        "metrics_available": trainers.metrics_for(setup.task_type),
        "tuning": {
            **request.tuning.model_dump(),
            "metric": tuning_metric,
            "tunable": [
                spec.name for spec in specs if _is_tunable(spec, request.tuning, setup.task_type)
            ],
        },
        "hyperparameters": {
            spec.name: trainers.resolve_hyperparameters(
                spec,
                setup.task_type,
                seed=seed,
                overrides=request.hyperparameters.get(spec.name),
            )
            for spec in specs
        },
    }


# -- the leaderboard ---------------------------------------------------------


def _training_matrix(frame: Any, setup: Any, pipeline: Preprocessor) -> TrainingMatrix:
    """The training split, and only the training split."""
    matrix = TrainingMatrix.from_split(frame=frame, spec=pipeline.to_dict(), setup=setup)
    trainers.assert_held_out(matrix.rows, setup.split.test)
    return matrix


def _score(
    setup: Any,
    pipeline: Preprocessor,
    frame: Any,
    fitted: trainers.FittedModel,
) -> dict[str, Any]:
    """Score one fitted Model on the held-out split.

    The **only** place the test matrix is built, and it runs after every Model
    has been tuned and fitted.

    The held-out truth is encoded against ``fitted.classes`` — the vocabulary the
    Model was *fitted* on — not one rebuilt from the test rows. A class that
    lives only in the training split (any Target class of a single row) would
    make a rebuilt vocabulary shorter, shifting every code after it and scoring
    the Model against the wrong truth without any error.
    """
    test_frame = frame.iloc[list(setup.split.test)]
    test_matrix = transform_with_spec(test_frame, pipeline.to_dict())
    if setup.task_type == trainers.CLASSIFICATION:
        y_true = trainers.encode_with_classes(
            test_frame[setup.target].to_numpy(), fitted.classes
        )
        return trainers.evaluate_classification(
            y_true,
            fitted.predict(test_matrix),
            proba=fitted.proba(test_matrix),
            classes=fitted.classes,
        )
    y_true = trainers.encode_regression(test_frame[setup.target].to_numpy())
    return trainers.evaluate_regression(y_true, fitted.predict(test_matrix))


def _error_entry(model: str, exc: Exception) -> dict[str, Any]:
    return {
        "model": model,
        "status": "failed",
        "error": f"{type(exc).__name__}: {exc}",
        "metrics": {},
    }


# -- the job -----------------------------------------------------------------


def run_training(
    ctx: JobContext,
    *,
    store: DatasetStore,
    checks: CheckStore,
    review_threshold: float = DEFAULT_REVIEW_THRESHOLD,
) -> dict[str, Any]:
    """Tune the selected Models on the training split, fit them, rank them.

    Resumable: the tuned hyperparameters and every finished Model are
    checkpointed, so an interrupted run picks up instead of re-searching.
    """
    request = TrainingRunRequest.model_validate(ctx.params)
    run_id = ctx.job_id
    stage = 0

    def report(stage_name: str, **extra: Any) -> None:
        ctx.progress(stage, len(STAGES), stage=stage_name, **extra)
        if ctx.cancelled:
            raise RuntimeError("cancelled")

    # -- setup: the plan and split from #31 ----------------------------------
    report("setup")
    version = store.get_version(request.version_id)
    frame = store.load_dataframe(request.version_id, include_provenance=True)
    setup_request = split_request(request)
    resolved = build_setup(setup_request, frame, review_threshold=review_threshold)
    setup = resolved.setup
    kept = resolved.frame
    seed = effective_seed(setup_request)
    plan = plan_training_run(request, frame, review_threshold=review_threshold)
    specs = trainers.resolve_models(request.models, setup.task_type)
    primary_metric = plan["primary_metric"]
    pipeline = _pipeline_for(store, request, kept, setup)
    stage = 1

    # -- the training matrix, and nothing else -------------------------------
    matrix = _training_matrix(kept, setup, pipeline)
    training = matrix.to_dict()
    report("matrix")
    stage = 2

    # -- tune: TrainingMatrix in, hyperparameters out ------------------------
    previous = ctx.load_checkpoint() or {}
    tuning: dict[str, Any] = {**request.tuning.model_dump(), "metric": plan["tuning"]["metric"], "results": {}}
    hyperparameters: dict[str, Any] = {
        name: dict(values) for name, values in plan["hyperparameters"].items()
    }
    if request.tuning.enabled:
        already = (previous.get("tuning") or {}).get("results") or {}
        for spec in specs:
            if spec.name in already:
                tuning["results"][spec.name] = already[spec.name]
            else:
                try:
                    tuning["results"][spec.name] = trainers.tune_model(
                        spec,
                        matrix,
                        metric=plan["tuning"]["metric"],
                        strategy=request.tuning.strategy,
                        n_iter=request.tuning.n_iter,
                        cv=request.tuning.cv,
                        search_space=request.tuning.search_space.get(spec.name),
                        overrides=request.hyperparameters.get(spec.name),
                        seed=seed,
                    )
                except Exception as exc:  # noqa: BLE001 - one bad search must not sink the run
                    tuning["results"][spec.name] = {
                        "model": spec.name,
                        "skipped": f"{type(exc).__name__}: {exc}",
                        "trained_on_rows": list(matrix.rows),
                    }
            best = (tuning["results"][spec.name] or {}).get("best_hyperparameters") or {}
            if best:
                hyperparameters[spec.name] = {**hyperparameters[spec.name], **best}
            report("tune", model=spec.name)
            _save(ctx, run_id, setup, pipeline, training, tuning, hyperparameters, [])
    stage = 3

    # -- fit: every selected Model, with the hyperparameters it ended up with -
    done = {
        entry["model"]: entry
        for entry in (previous.get("leaderboard") or [])
        if entry.get("status") == "ok"
        and entry.get("hyperparameters") == hyperparameters.get(entry.get("model"))
    }
    fitted_models: dict[str, trainers.FittedModel] = {}
    entries: list[dict[str, Any]] = []
    for spec in specs:
        if spec.name in done:
            entries.append(done[spec.name])
            continue
        try:
            fitted = trainers.fit_model(
                spec,
                setup.task_type,
                matrix.X,
                matrix.y,
                hyperparameters=hyperparameters[spec.name],
                classes=matrix.classes,
            )
        except Exception as exc:  # noqa: BLE001 - one bad Model must not sink the board
            entries.append(_error_entry(spec.name, exc))
        else:
            fitted_models[spec.name] = fitted
            entries.append({**fitted.to_dict(), "status": "ok", "metrics": {}})
        report("fit", model=spec.name)
        _save(ctx, run_id, setup, pipeline, training, tuning, hyperparameters, entries)
    stage = 4

    # -- score: the held-out test split, once everything else is done ---------
    for entry in entries:
        if entry.get("status") != "ok":
            continue
        if entry.get("metrics"):
            continue  # a resumed run: this Model is already scored
        report("score", model=entry["model"])
        spec = MODEL_SPECS[entry["model"]]
        try:
            # A Model fitted in *this* pass is still in memory; one restored from
            # a checkpoint has to be refitted before it can be scored.
            fitted = fitted_models.get(spec.name) or trainers.fit_model(
                spec,
                setup.task_type,
                matrix.X,
                matrix.y,
                hyperparameters=entry["hyperparameters"],
                classes=matrix.classes,
            )
            entry["metrics"] = _score(setup, pipeline, kept, fitted)
        except Exception as exc:  # noqa: BLE001 - report it, do not lose the other Models
            entry["status"] = "failed"
            entry["error"] = f"{type(exc).__name__}: {exc}"
    stage = 5

    leaderboard = trainers.rank_leaderboard(entries, primary_metric)
    warnings = trainers.leaderboard_warnings(leaderboard, primary_metric)
    if not any(entry.get("rank") is not None for entry in leaderboard):
        warnings.append(
            f"no Model produced {primary_metric!r}; a {setup.task_type} Target can be "
            f"ranked on: {', '.join(trainers.metrics_for(setup.task_type))}"
        )
    stage = 6

    # -- Checks: the same vocabulary #31 raises -------------------------------
    train_frame = kept.iloc[list(setup.split.train)]
    report_checks = evaluate_training_checks(
        setup=setup,
        train_frame=train_frame,
        test_frame=kept.iloc[list(setup.split.test)],
        matrix=matrix.X,
        values=train_frame[setup.target].to_numpy(),
        sources=pipeline.spec.output_sources,
        feature_names=pipeline.feature_names,
    )
    raised = raise_training_checks(checks, run_id, report_checks, step="train")
    report("checks")

    run = {
        "training_run_id": run_id,
        "project_id": version.project_id,
        "version_id": setup.version_id,
        "target": setup.target,
        "task_type": setup.task_type,
        "label_family": label_family(setup.target),
        "seed": seed,
        "seeds": {"split": setup.seed, "models": seed},
        "library_versions": trainers.library_versions(
            sorted({spec.library for spec in specs})
        ),
        "primary_metric": primary_metric,
        "primary_metric_label": METRICS[primary_metric].label,
        "primary_metric_higher_is_better": METRICS[primary_metric].higher_is_better,
        "metrics_available": trainers.metrics_for(setup.task_type),
        "models": [spec.to_dict() for spec in specs],
        "unavailable_models": plan["unavailable_models"],
        "hyperparameters": hyperparameters,
        "tuning": tuning,
        "training_split": training,
        "test_split": {
            "rows": len(setup.split.test),
            "indices": [int(i) for i in setup.split.test],
            "stratified": bool(setup.split.stratified),
            "test_size": float(setup.test_size),
        },
        "feature_names": pipeline.feature_names,
        "n_features": pipeline.n_features,
        "setup": setup.to_dict(),
        "preprocessing": pipeline.to_dict(),
        "leaderboard": leaderboard,
        "warnings": warnings,
        "checks": report_checks,
        "checks_raised": raised,
    }
    _save(
        ctx,
        run_id,
        setup,
        pipeline,
        training,
        tuning,
        hyperparameters,
        leaderboard,
        run=run,
    )
    stage = 7
    report("done")
    return run


def _pipeline_for(
    store: DatasetStore, request: TrainingRunRequest, kept: Any, setup: Any
) -> Preprocessor:
    """The preprocessing for this Training Run, fitted on the training split.

    When the request names a ``setup_run_id``, that run's checkpointed spec is
    reused verbatim (and applied downstream with ``transform_with_spec``); the
    recorded split must match the one this run resolved, or it is not the run
    the user reviewed.
    """
    if request.setup_run_id:
        checkpoint = _load_setup_checkpoint(store, request.setup_run_id)
        if not checkpoint:
            raise ModelError(
                f"setup run {request.setup_run_id!r} has no checkpoint to reuse; run "
                "POST /api/train/setup first, or drop setup_run_id"
            )
        recorded = (checkpoint.get("setup") or {}).get("split") or {}
        if recorded and int(recorded.get("test_rows", -1)) != len(setup.split.test):
            raise ModelError(
                f"setup run {request.setup_run_id!r} split the rows differently from this "
                "Training Run; re-run the setup or drop setup_run_id"
            )
        if list(checkpoint.get("test_indices") or []) != [int(i) for i in setup.split.test]:
            raise ModelError(
                f"setup run {request.setup_run_id!r} held out different rows than this "
                "Training Run; re-run the setup or drop setup_run_id"
            )
        return Preprocessor.from_dict(checkpoint.get("preprocessing") or {})
    return fit_preprocessor(
        kept.iloc[list(setup.split.train)],
        features=setup.feature_columns,
        ngram_range=request.ngram_range,
        min_df=request.min_df,
        max_features=request.max_features,
        scale=request.scale,
    )


def _load_setup_checkpoint(store: DatasetStore, run_id: str) -> dict[str, Any] | None:
    """Read another job's checkpoint through the store's database."""
    with store.db.connect() as conn:
        row = conn.execute(
            "SELECT checkpoint_json FROM jobs WHERE id = ? AND type = 'train_setup'", (run_id,)
        ).fetchone()
    if row is None or not row["checkpoint_json"]:
        return None
    return json.loads(row["checkpoint_json"])


def _save(
    ctx: JobContext,
    run_id: str,
    setup: Any,
    pipeline: Preprocessor,
    training: Mapping[str, Any],
    tuning: Mapping[str, Any],
    hyperparameters: Mapping[str, Any],
    leaderboard: Sequence[Mapping[str, Any]],
    *,
    run: dict[str, Any] | None = None,
) -> None:
    """Checkpoint the Training Run: the plan, the split, the search, the board."""
    payload: dict[str, Any] = {
        "training_run_id": run_id,
        "setup": setup.to_dict(),
        "preprocessing": pipeline.to_dict(),
        "train_indices": [int(i) for i in training.get("rows") or []],
        "test_indices": [int(i) for i in setup.split.test],
        "training_split": dict(training),
        "hyperparameters": dict(hyperparameters),
        "tuning": dict(tuning),
        "leaderboard": [dict(entry) for entry in leaderboard],
    }
    if run is not None:
        payload["run"] = run
    ctx.save_checkpoint(payload)


def register_training_run_job(manager: Any, app: Any) -> None:
    """Wire the ``train`` job type to the app's stores.

    Resumable: the checkpoint holds the tuned hyperparameters and every finished
    Model, so an interrupted run resumes instead of re-searching and re-fitting.
    """

    def runner(ctx: JobContext) -> dict[str, Any]:
        review_threshold = float(
            app.state.settings.load().get("review_threshold", DEFAULT_REVIEW_THRESHOLD)
        )
        return run_training(
            ctx,
            store=app.state.store,
            checks=app.state.checks,
            review_threshold=review_threshold,
        )

    manager.register("train", runner, resumable=True)
