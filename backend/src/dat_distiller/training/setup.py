"""Everything a Training Run needs before a Model is fitted.

:func:`build_setup` turns a request into a resolved, serializable
:class:`TrainingSetup` — Target, Task Type, the feature columns that survived
the leakage guard, the held-out split and the row accounting — and
:func:`run_setup` adds the fitted preprocessing pipeline, raises the Checks and
checkpoints all of it, so the leaderboard (#32) and the Model Bundle (#34) pick
up exactly this work.

The pipeline is fitted on the **training split only**; the held-out test split
never influences it, and no tuning sees it.
"""

from __future__ import annotations

import json
import zlib
from dataclasses import dataclass, field
from typing import Any, Literal

import pandas as pd
from pydantic import BaseModel, Field, field_validator

from ..checks import CheckStore
from ..jobs import JobContext
from ..store import DatasetStore
from ..store.provenance import PROVENANCE_COLUMN
from .checks import evaluate_training_checks, raise_training_checks, value_key
from .preprocess import (
    DEFAULT_MAX_FEATURES,
    DEFAULT_MIN_DF,
    DEFAULT_NGRAM_RANGE,
    fit_preprocessor,
)
from .split import DEFAULT_TEST_SIZE, Split, split_indices
from .task import label_family, resolve_task_type, sibling_columns

#: Mirrors ``settings.review_threshold``: below this a Jev label needs review.
DEFAULT_REVIEW_THRESHOLD = 0.8

#: Why a column is not a feature.
REASON_TARGET = "target"
REASON_SIBLING = "target_sibling"
REASON_SENSITIVE = "sensitive_attribute"
REASON_USER = "user_excluded"
REASON_PROVENANCE = "provenance"

#: The stages the job reports progress over.
STAGES = ("prepare", "fit", "transform", "checks", "done")


class TrainingSetupRequest(BaseModel):
    """Body shared by estimate / validate / setup."""

    version_id: str
    target: str
    #: ``None`` means infer it from the Target's values.
    task_type: Literal["classification", "regression"] | None = None
    #: ``None`` means every column except the Target, the Provenance column and
    #: the auto-excluded ones; a list is an explicit allow-list.
    features: list[str] | None = None
    #: Removed on top of whatever the allow-list resolved to.
    exclude_features: list[str] = Field(default_factory=list)
    sensitive_attribute: str | None = None
    #: The Sensitive Attribute is left out of the features by default.
    include_sensitive_attribute: bool = False
    #: Drop the rows whose Jev label is below the review threshold.
    exclude_unreviewed: bool = False
    test_size: float = Field(default=DEFAULT_TEST_SIZE, gt=0.0, lt=1.0)
    #: ``None`` derives a stable seed from the request so a re-run reproduces it.
    seed: int | None = None
    ngram_range: list[int] = Field(default_factory=lambda: list(DEFAULT_NGRAM_RANGE), min_length=2, max_length=2)
    max_features: int = Field(default=DEFAULT_MAX_FEATURES, ge=1)
    min_df: int = Field(default=DEFAULT_MIN_DF, ge=1)
    scale: bool = True

    @field_validator("ngram_range")
    @classmethod
    def _check_ngram_range(cls, value: list[int]) -> list[int]:
        low, high = value
        if low < 1 or high < low:
            raise ValueError("ngram_range must be 1 <= low <= high")
        return [low, high]

    def to_params(self) -> dict[str, Any]:
        return self.model_dump()


def effective_seed(request: TrainingSetupRequest) -> int:
    """Explicit seed wins; otherwise a stable hash of the request, so re-running
    an unseeded Training Run reproduces the same split."""
    if request.seed is not None:
        return int(request.seed)
    payload = request.model_dump(exclude={"seed"})
    for key in ("features", "exclude_features"):
        if isinstance(payload.get(key), list):
            payload[key] = sorted(payload[key])
    canonical = json.dumps(payload, sort_keys=True, default=str)
    return zlib.crc32(canonical.encode()) & 0x7FFFFFFF


@dataclass
class TrainingSetup:
    """The resolved plan for one Training Run. ``to_dict()`` is what is persisted."""

    version_id: str
    target: str
    task_type: str
    feature_columns: list[str]
    excluded_columns: dict[str, str]
    seed: int
    test_size: float
    split: Split
    label_family: str
    siblings: list[str] = field(default_factory=list)
    sensitive_attribute: str | None = None
    include_sensitive_attribute: bool = False
    exclude_unreviewed: bool = False
    review_threshold: float = DEFAULT_REVIEW_THRESHOLD
    confidence_column: str | None = None
    kept_rows: int = 0
    dropped_null_target: int = 0
    dropped_unreviewed: int = 0
    unreviewed_rows: int = 0
    reviewed_rows: int = 0
    class_counts: dict[str, int] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "version_id": self.version_id,
            "target": self.target,
            "task_type": self.task_type,
            "feature_columns": list(self.feature_columns),
            "excluded_columns": dict(self.excluded_columns),
            "seed": int(self.seed),
            "test_size": float(self.test_size),
            "split": self.split.to_dict(),
            "label_family": self.label_family,
            "siblings": list(self.siblings),
            "sensitive_attribute": self.sensitive_attribute,
            "include_sensitive_attribute": bool(self.include_sensitive_attribute),
            "exclude_unreviewed": bool(self.exclude_unreviewed),
            "review_threshold": float(self.review_threshold),
            "confidence_column": self.confidence_column,
            "kept_rows": int(self.kept_rows),
            "dropped_null_target": int(self.dropped_null_target),
            "dropped_unreviewed": int(self.dropped_unreviewed),
            "unreviewed_rows": int(self.unreviewed_rows),
            "reviewed_rows": int(self.reviewed_rows),
            "class_counts": dict(self.class_counts),
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> TrainingSetup:
        """Read a persisted plan back.

        The split's *indices* are not part of ``to_dict()`` — they travel
        separately in the job checkpoint, next to the preprocessing spec — so
        the restored plan knows the row counts but has no rows attached.
        """
        return cls(
            version_id=str(payload["version_id"]),
            target=str(payload["target"]),
            task_type=str(payload["task_type"]),
            feature_columns=list(payload.get("feature_columns") or []),
            excluded_columns=dict(payload.get("excluded_columns") or {}),
            seed=int(payload.get("seed", 0)),
            test_size=float(payload.get("test_size", DEFAULT_TEST_SIZE)),
            split=Split.from_dict(payload.get("split") or {}),
            label_family=str(payload.get("label_family") or payload["target"]),
            siblings=list(payload.get("siblings") or []),
            sensitive_attribute=payload.get("sensitive_attribute"),
            include_sensitive_attribute=bool(payload.get("include_sensitive_attribute", False)),
            exclude_unreviewed=bool(payload.get("exclude_unreviewed", False)),
            review_threshold=float(payload.get("review_threshold", DEFAULT_REVIEW_THRESHOLD)),
            confidence_column=payload.get("confidence_column"),
            kept_rows=int(payload.get("kept_rows", 0)),
            dropped_null_target=int(payload.get("dropped_null_target", 0)),
            dropped_unreviewed=int(payload.get("dropped_unreviewed", 0)),
            unreviewed_rows=int(payload.get("unreviewed_rows", 0)),
            reviewed_rows=int(payload.get("reviewed_rows", 0)),
            class_counts=dict(payload.get("class_counts") or {}),
        )


@dataclass
class ResolvedSetup:
    """A :class:`TrainingSetup` plus the rows it applies to, in split order."""

    setup: TrainingSetup
    frame: pd.DataFrame

    def train_frame(self) -> pd.DataFrame:
        return self.frame.iloc[self.setup.split.train]

    def test_frame(self) -> pd.DataFrame:
        return self.frame.iloc[self.setup.split.test]


# -- unreviewed rows --------------------------------------------------------


def _confidence_column(frame: pd.DataFrame, target: str) -> str | None:
    column = f"{label_family(target)}__confidence"
    return column if column in frame.columns else None


def _label_origin(prov: Any, family: str) -> str | None:
    label_origins = prov.get("label_origins") or {}
    entry = label_origins.get(family)
    if entry is None:
        return None
    return str(entry.get("origin") or "")


def unreviewed_mask(
    frame: pd.DataFrame, target: str, *, review_threshold: float = DEFAULT_REVIEW_THRESHOLD
) -> pd.Series:
    """Rows whose Target came from Jev below the review threshold and was never
    looked at by a human — the rows the Review Queue is holding."""
    mask = pd.Series(False, index=frame.index)
    confidence_column = _confidence_column(frame, target)
    if confidence_column is None or PROVENANCE_COLUMN not in frame.columns:
        return mask
    family = label_family(target)
    confidences = pd.to_numeric(frame[confidence_column], errors="coerce")
    for row_index in frame.index:
        try:
            prov = json.loads(frame.at[row_index, PROVENANCE_COLUMN])
        except (TypeError, ValueError):
            continue
        if not isinstance(prov, dict):
            continue
        if _label_origin(prov, family) != "jev":
            continue
        confidence = confidences.at[row_index]
        if pd.isna(confidence) or float(confidence) < review_threshold:
            mask.at[row_index] = True
    return mask


# -- building the plan ------------------------------------------------------


def _exclusion_reason(
    column: str,
    *,
    target: str,
    siblings: set[str],
    excluded_by_user: set[str],
    sensitive_attribute: str | None,
    include_sensitive_attribute: bool,
) -> str | None:
    if column == PROVENANCE_COLUMN:
        return REASON_PROVENANCE
    if column == target:
        return REASON_TARGET
    if column in siblings:
        return REASON_SIBLING
    if column in excluded_by_user:
        return REASON_USER
    if column == sensitive_attribute and not include_sensitive_attribute:
        return REASON_SENSITIVE
    return None


def build_setup(
    request: TrainingSetupRequest,
    frame: pd.DataFrame,
    *,
    review_threshold: float = DEFAULT_REVIEW_THRESHOLD,
) -> ResolvedSetup:
    """Resolve a request against a Dataset Version: features, rows and split."""
    columns = [str(c) for c in frame.columns]
    if request.target not in columns:
        raise ValueError(f"unknown Target {request.target!r}; this Version has {columns}")
    if request.sensitive_attribute is not None:
        if request.sensitive_attribute not in columns:
            raise ValueError(f"unknown Sensitive Attribute {request.sensitive_attribute!r}")
        if request.sensitive_attribute == request.target:
            raise ValueError("the Sensitive Attribute cannot also be the Target")

    candidates = (
        list(request.features)
        if request.features is not None
        else [c for c in columns if c not in (request.target, PROVENANCE_COLUMN)]
    )
    unknown = [c for c in candidates if c not in columns]
    if unknown:
        raise ValueError(f"unknown feature columns: {sorted(unknown)}")
    unknown_excluded = [c for c in request.exclude_features if c not in columns]
    if unknown_excluded:
        raise ValueError(f"unknown excluded columns: {sorted(unknown_excluded)}")

    excluded_by_user = set(request.exclude_features)
    siblings = set(sibling_columns(request.target, columns))
    excluded: dict[str, str] = {}
    for column in columns:
        reason = _exclusion_reason(
            column,
            target=request.target,
            siblings=siblings,
            excluded_by_user=excluded_by_user,
            sensitive_attribute=request.sensitive_attribute,
            include_sensitive_attribute=request.include_sensitive_attribute,
        )
        if reason is not None:
            excluded[column] = reason
    feature_columns = [c for c in candidates if c not in excluded]
    if not feature_columns:
        raise ValueError("no feature columns left for this Training Run")

    task_type = resolve_task_type(frame[request.target], request.task_type)

    working = frame[frame[request.target].notna()].reset_index(drop=True)
    dropped_null_target = len(frame) - len(working)
    if len(working) < 2:
        raise ValueError("a Training Run needs at least 2 rows with a Target value")
    if int(working[request.target].nunique()) < 2:
        raise ValueError(
            f"Target {request.target!r} has fewer than 2 distinct values; nothing to learn"
        )

    mask = unreviewed_mask(working, request.target, review_threshold=review_threshold)
    unreviewed_total = int(mask.sum())
    if request.exclude_unreviewed:
        kept = working.loc[~mask]
        dropped_unreviewed = unreviewed_total
        unreviewed_rows = 0  # none of them reach the split
    else:
        kept = working
        dropped_unreviewed = 0
        unreviewed_rows = unreviewed_total
    if len(kept) < 2:
        raise ValueError(
            "excluding unreviewed low-confidence rows leaves fewer than 2 rows; "
            "review them or turn the option off"
        )

    target_values = kept[request.target]
    distinct = target_values.nunique()
    if int(distinct) < 2:
        raise ValueError(
            f"Target {request.target!r} has fewer than 2 distinct values left; nothing to learn"
        )
    class_counts = (
        {value_key(value): int(count) for value, count in target_values.value_counts().items()}
        if task_type == "classification"
        else {}
    )

    seed = effective_seed(request)
    split = split_indices(
        len(kept),
        test_size=request.test_size,
        seed=seed,
        stratify=target_values.to_numpy() if task_type == "classification" else None,
    )

    setup = TrainingSetup(
        version_id=request.version_id,
        target=request.target,
        task_type=task_type,
        feature_columns=feature_columns,
        excluded_columns=excluded,
        seed=seed,
        test_size=request.test_size,
        split=split,
        label_family=label_family(request.target),
        siblings=sorted(siblings),
        sensitive_attribute=request.sensitive_attribute,
        include_sensitive_attribute=request.include_sensitive_attribute,
        exclude_unreviewed=request.exclude_unreviewed,
        review_threshold=review_threshold,
        confidence_column=_confidence_column(kept, request.target),
        kept_rows=len(kept),
        dropped_null_target=dropped_null_target,
        dropped_unreviewed=dropped_unreviewed,
        unreviewed_rows=unreviewed_rows,
        reviewed_rows=len(working) - unreviewed_total,
        class_counts=class_counts,
    )
    return ResolvedSetup(setup=setup, frame=kept.reset_index(drop=True))


# -- estimate ---------------------------------------------------------------


def estimate_setup(
    request: TrainingSetupRequest,
    frame: pd.DataFrame,
    *,
    review_threshold: float = DEFAULT_REVIEW_THRESHOLD,
) -> dict[str, Any]:
    """The plan before anything is fitted: rows, features, split, seed."""
    resolved = build_setup(request, frame, review_threshold=review_threshold)
    setup = resolved.setup
    return {
        "version_id": setup.version_id,
        "target": setup.target,
        "task_type": setup.task_type,
        "label_family": setup.label_family,
        "rows": len(frame),
        "kept_rows": setup.kept_rows,
        "train_rows": len(setup.split.train),
        "test_rows": len(setup.split.test),
        "test_size": setup.test_size,
        "stratified": setup.split.stratified,
        "seed": setup.seed,
        "n_source_features": len(setup.feature_columns),
        "feature_columns": list(setup.feature_columns),
        "excluded_columns": dict(setup.excluded_columns),
        "sensitive_attribute": setup.sensitive_attribute,
        "sensitive_attribute_included": setup.include_sensitive_attribute,
        "exclude_unreviewed": setup.exclude_unreviewed,
        "class_counts": dict(setup.class_counts),
        "unreviewed": {
            "family": setup.label_family,
            "confidence_column": setup.confidence_column,
            "threshold": setup.review_threshold,
            "unreviewed_rows": setup.unreviewed_rows,
            "excluded_rows": setup.dropped_unreviewed,
            "reviewed_rows": setup.reviewed_rows,
        },
    }


# -- the run ----------------------------------------------------------------


def run_setup(
    ctx: JobContext,
    *,
    store: DatasetStore,
    checks: CheckStore,
    review_threshold: float = DEFAULT_REVIEW_THRESHOLD,
) -> dict[str, Any]:
    """Resolve, preprocess, split and Check one Training Run.

    The pipeline is fitted on the training split only; everything it needs to be
    reproduced is checkpointed on the job.
    """
    request = TrainingSetupRequest.model_validate(ctx.params)
    run_id = ctx.job_id
    stage = 0

    def progress(stage_name: str) -> None:
        nonlocal stage
        ctx.progress(stage, len(STAGES), stage=stage_name)
        if ctx.cancelled:
            raise RuntimeError("cancelled")

    progress("prepare")
    version = store.get_version(request.version_id)
    frame = store.load_dataframe(request.version_id, include_provenance=True)
    resolved = build_setup(request, frame, review_threshold=review_threshold)
    setup = resolved.setup
    stage = 1

    train_frame = resolved.train_frame()
    pipeline = fit_preprocessor(
        train_frame,
        features=setup.feature_columns,
        ngram_range=request.ngram_range,
        min_df=request.min_df,
        max_features=request.max_features,
        scale=request.scale,
    )
    progress("fit")
    stage = 2

    matrix = pipeline.transform(train_frame)
    values = train_frame[setup.target].to_numpy()
    test_frame = resolved.test_frame()
    progress("transform")
    stage = 3

    report = evaluate_training_checks(
        setup=setup,
        train_frame=train_frame,
        test_frame=test_frame,
        matrix=matrix,
        values=values,
        sources=pipeline.spec.output_sources,
        feature_names=pipeline.feature_names,
    )
    raised = raise_training_checks(checks, run_id, report)
    progress("checks")
    stage = 4

    spec = pipeline.to_dict()
    ctx.save_checkpoint(
        {
            "training_run_id": run_id,
            "setup": setup.to_dict(),
            "preprocessing": spec,
            "checks": report,
            "train_indices": [int(i) for i in setup.split.train],
            "test_indices": [int(i) for i in setup.split.test],
        }
    )
    progress("done")
    return {
        "training_run_id": run_id,
        "version_id": setup.version_id,
        "project_id": version.project_id,
        "target": setup.target,
        "task_type": setup.task_type,
        "label_family": setup.label_family,
        "feature_columns": list(setup.feature_columns),
        "excluded_columns": dict(setup.excluded_columns),
        "feature_names": pipeline.feature_names,
        "n_features": pipeline.n_features,
        "sensitive_attribute": setup.sensitive_attribute,
        "sensitive_attribute_included": setup.include_sensitive_attribute,
        "exclude_unreviewed": setup.exclude_unreviewed,
        "seed": setup.seed,
        "test_size": setup.test_size,
        "stratified": setup.split.stratified,
        "rows": len(frame),
        "kept_rows": setup.kept_rows,
        "train_rows": len(train_frame),
        "test_rows": len(test_frame),
        "dropped_null_target": setup.dropped_null_target,
        "dropped_unreviewed": setup.dropped_unreviewed,
        "unreviewed_rows": setup.unreviewed_rows,
        "reviewed_rows": setup.reviewed_rows,
        "class_counts": dict(setup.class_counts),
        "preprocessing": spec,
        "checks_raised": raised,
        "checks": report,
    }


def register_training_job(manager: Any, app: Any) -> None:
    """Wire the ``train_setup`` job type to the app's stores."""

    def runner(ctx: JobContext) -> dict[str, Any]:
        review_threshold = float(
            app.state.settings.load().get("review_threshold", DEFAULT_REVIEW_THRESHOLD)
        )
        return run_setup(
            ctx,
            store=app.state.store,
            checks=app.state.checks,
            review_threshold=review_threshold,
        )

    manager.register("train_setup", runner)
