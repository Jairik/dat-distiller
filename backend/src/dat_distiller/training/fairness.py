"""The **Fairness Report**: per-group metrics for a Model, split by a Sensitive Attribute.

Computed **on demand** — nothing here runs at training time, and a Training Run
that was never asked for a Fairness Report behaves exactly as it did before. The
caller picks a Model, a Sensitive Attribute and a threshold, and gets back a
report plus the warning **Checks** its gaps earned.

Three decisions this module makes, on purpose:

1. **It reuses the stored pipeline, it does not rebuild one.** The report reads
   the Training Run's own ``preprocessing`` spec, its own ``test_indices`` and
   its own per-Model ``hyperparameters`` out of the persisted run, re-derives the
   split from the run's own request (the split is a pure function of the request
   and its seed), and then *verifies* that the split it just derived is
   identical to the one the run recorded — :func:`load_run_context` raises
   :class:`FairnessError` when it is not. So a Fairness Report can never be
   computed against different rows, a different matrix, or a differently
   preprocessed Model than the leaderboard was.

2. **It refits the Model rather than persisting an estimator.** A persisted
   pickle is a version-locked liability, and the fit is fully determined by data
   the run already stores (split + preprocessing + hyperparameters, seed
   included), so the refit reproduces the same estimator. The report records
   ``refit: true`` so nobody mistakes a refit for the original object.

3. **A group that cannot be measured says so.** The group list comes from the
   whole Dataset Version, not from the test split, so a group with zero held-out
   rows, a group dropped before the split, a group with no positive cases (TPR
   undefined) and a group with no negative cases (FPR undefined) all appear with
   a ``null`` metric and a reason. A gap is computed only over the groups that
   *have* a value, and says how many it used and which it could not.

Which gap gets flagged, and against what:

- **Classification** — per-group accuracy, TPR, FPR and selection rate. The
  selection-rate gap is the **demographic parity difference**; TPR, FPR and
  accuracy are the other three. All four are rates, so all four are comparable
  with the configured threshold.
- **Regression** — per-group MAE (plus RMSE and mean error for context). The MAE
  gap is in the Target's *own units*, so comparing it with a rate threshold would
  be meaningless: the absolute gap is reported with ``exceeds: null`` and a
  reason, and the gap that is compared is the **relative** one — the gap as a
  share of the Model's overall MAE.
- The **positive class** is the last class in the Target's sorted class order
  (:func:`trainers.encode_classes` sorts, so ``False``/``0`` is negative and
  ``True``/``1`` is positive), and the caller can name another.
- When several gaps clear the threshold each gets its own Check naming its own
  groups and numbers, and the report's ``gaps_exceeding_threshold`` is ordered
  widest first with ``largest_gap`` naming the widest gap overall — so "which gap
  is the finding" is never a guess.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from ..extras import is_installed
from .checks import value_key
from .run_rows import RunContext, resolve_run
from .trainers import (
    CLASSIFICATION,
    MODEL_SPECS,
    REGRESSION,
    ModelError,
    encode_regression,
    encode_with_classes as _encode_with_classes,
    fit_model,
    get_spec,
)

#: Mirrors ``settings.fairness_gap_threshold``: a gap wider than this is a Check.
DEFAULT_GAP_THRESHOLD = 0.1

#: The Check kind for a gap that clears the threshold.
FAIRNESS_GAP_CHECK_KIND = "fairness_gap"
#: The Check kind for a Fairness Report on a Sensitive Attribute the Model could see.
FAIRNESS_USED_CHECK_KIND = "fairness_sensitive_attribute_used"
#: ``step`` recorded on both, so the UI groups them with the Train step.
FAIRNESS_STEP = "fairness"

#: The group metric names per Task Type.
CLASSIFICATION_METRICS = ("accuracy", "selection_rate", "tpr", "fpr")
REGRESSION_METRICS = ("mae", "rmse", "mean_error")

#: Unit of a gap: a rate (comparable with the threshold) or the Target's own units.
UNIT_RATE = "rate"
UNIT_TARGET = "target_units"

#: The group key used for missing values, so a missing group is visible, not absent.
MISSING_GROUP = "missing"

_ROUND = 6
#: Guards a division by an overall MAE of exactly zero.
_TINY = 1e-12


class FairnessError(ValueError):
    """A Fairness Report this Training Run cannot honestly produce."""


# -- the threshold -----------------------------------------------------------


def resolve_gap_threshold(value: Any, default: float = DEFAULT_GAP_THRESHOLD) -> float:
    """The caller's threshold, or the configured one, validated.

    It is compared with rate gaps, so it must be a finite number in ``[0, 1]``.
    A threshold outside that range would make every gap either trivially zero or
    trivially over, which is why it is refused rather than clamped.
    """
    threshold = default if value is None else float(value)
    if not math.isfinite(threshold):
        raise FairnessError(f"fairness gap threshold must be a finite number, got {value!r}")
    if not 0.0 <= threshold <= 1.0:
        raise FairnessError(
            f"fairness gap threshold must be between 0 and 1, got {threshold}; it is compared "
            "with differences between rates"
        )
    return threshold


# -- the positive class ------------------------------------------------------


def _label_keys(label: Any) -> list[str]:
    """The class keys a user-supplied label could mean, most likely first.

    A label arrives from JSON as a string, and ``value_key("True")`` is
    ``str:True`` while ``value_key(True)`` is ``bool:True`` — so the literal text
    is tried next, then the obvious coercions. Every candidate is an *existing*
    class or is discarded by the caller; nothing is invented.
    """
    candidates = [value_key(label)]
    text = str(label).strip()
    for extra in (text,):
        if extra not in candidates:
            candidates.append(extra)
    lowered = text.lower()
    for literal, coerced in (("true", True), ("false", False), ("none", None)):
        if lowered == literal:
            key = value_key(coerced)
            if key not in candidates:
                candidates.append(key)
    try:
        number = float(text)
    except (TypeError, ValueError):
        pass
    else:
        key = value_key(number)
        if key not in candidates:
            candidates.append(key)
    return candidates


def positive_class_code(classes: Sequence[str], positive_label: Any = None) -> tuple[int, str]:
    """Which class code counts as "positive" for TPR, FPR and selection rate.

    With no label named, the **last** class in the sorted class order is positive
    — the same convention the leaderboard's ROC-AUC uses — so ``False``/``0`` is
    negative and ``True``/``1`` is positive. A named label is resolved against
    the class keys the Target actually has, and one that matches nothing is an
    error rather than a silent fallback to the default.
    """
    names = [str(c) for c in (classes or [])]
    if not names:
        raise FairnessError("this Target has no classes, so it has no positive class")
    if positive_label is None:
        return len(names) - 1, names[-1]
    for key in _label_keys(positive_label):
        if key in names:
            return names.index(key), key
    raise FairnessError(
        f"{positive_label!r} is not a class of this Target; its classes are: {', '.join(names)}"
    )


def encode_with_classes(values: Sequence[Any], classes: Sequence[str]) -> np.ndarray:
    """Class-encode values against a **fixed** class list.

    The mapping itself lives in :mod:`.trainers`, because it is the *Model's*
    numbering that has to be respected, and the leaderboard and a Fairness
    Report must agree on it. This wrapper only re-labels the refusal, so a
    report can still say so in its own voice.
    """
    try:
        return _encode_with_classes(values, classes)
    except ModelError as exc:
        raise FairnessError(str(exc)) from exc


# -- per-group metrics -------------------------------------------------------


def _measured(value: Any) -> dict[str, Any]:
    number = float(value)
    if not math.isfinite(number):
        return {"value": None, "reason": "the computation was not finite"}
    return {"value": round(number, _ROUND), "reason": None}


def _missing(reason: str) -> dict[str, Any]:
    return {"value": None, "reason": reason}


def group_sizes(values: Sequence[Any]) -> dict[str, int]:
    """``value_key`` -> row count, sorted by key so the report order is stable."""
    counts: dict[str, int] = {}
    for value in values:
        key = value_key(value)
        counts[key] = counts.get(key, 0) + 1
    return {key: counts[key] for key in sorted(counts)}


def group_keys(values: Sequence[Any]) -> list[str]:
    """``value_key`` per value, in row order."""
    return [value_key(value) for value in values]


def _absent_reason(key: str, kept: int, seen: int) -> str:
    if kept == 0:
        return (
            f"group {key!r} has no rows among the rows this Training Run kept; all {seen} of its "
            "row(s) in this Dataset Version were dropped before the split"
        )
    return (
        f"no rows of group {key!r} are in the held-out test split "
        f"({kept} row(s) in this Training Run's rows, all of them in training)"
    )


def _unmeasured_group(
    key: str, *, metrics: Sequence[str], kept: int, seen: int, reason: str
) -> dict[str, Any]:
    """A group that is present in the data and cannot be measured."""
    return {
        "group": key,
        "n_version": int(seen),
        "n_total": int(kept),
        "n_train": int(kept),
        "n_test": 0,
        "n_actual_positive": None,
        "n_actual_negative": None,
        "n_predicted_positive": 0,
        "class_counts": {},
        "measured": False,
        "reason": reason,
        "metrics": {name: _missing(reason) for name in metrics},
    }


def _row_mask(group_of_row: Sequence[str], key: str) -> np.ndarray:
    return np.fromiter((value == key for value in group_of_row), dtype=bool, count=len(group_of_row))


def _check_lengths(group_of_row: Sequence[str], truth: np.ndarray, predicted: np.ndarray) -> None:
    if not (len(group_of_row) == len(truth) == len(predicted)):
        raise FairnessError(
            "the group column, the Target and the predictions must have one entry per "
            f"held-out row; got {len(group_of_row)}, {len(truth)} and {len(predicted)}"
        )


def _all_groups(
    group_of_row: Sequence[str],
    kept_counts: Mapping[str, int],
    version_counts: Mapping[str, int],
) -> list[str]:
    """Every group in the Dataset Version, in stable key order.

    Taken from the **whole** version, not from the test split: a group that never
    reached the test split has to be reported as unmeasurable, which is only
    possible if it is in the list in the first place.
    """
    return sorted(set(kept_counts) | set(version_counts) | set(group_of_row))


def classification_groups(
    *,
    group_of_row: Sequence[str],
    y_true: Sequence[int],
    y_pred: Sequence[int],
    positive: int,
    kept_counts: Mapping[str, int],
    version_counts: Mapping[str, int] | None = None,
    classes: Sequence[str] | None = None,
) -> list[dict[str, Any]]:
    """Accuracy, TPR, FPR and selection rate for every group, in group order.

    ``group_of_row`` is one ``value_key`` per **test-split** row. TPR needs a
    positive case in the group and FPR needs a negative one; a group without
    either gets ``null`` for that one metric and keeps the rest.
    """
    truth = np.asarray(y_true, dtype="int64")
    predicted = np.asarray(y_pred, dtype="int64")
    _check_lengths(group_of_row, truth, predicted)
    names = [str(c) for c in (classes or [])]
    positive_name = names[positive] if 0 <= positive < len(names) else str(positive)
    version = dict(version_counts or {})
    rows: list[dict[str, Any]] = []
    for key in _all_groups(group_of_row, kept_counts, version):
        mask = _row_mask(group_of_row, key)
        n_test = int(mask.sum())
        kept = int(kept_counts.get(key, 0))
        seen = int(version.get(key, 0))
        if n_test == 0:
            rows.append(
                _unmeasured_group(
                    key,
                    metrics=CLASSIFICATION_METRICS,
                    kept=kept,
                    seen=seen,
                    reason=_absent_reason(key, kept, seen),
                )
            )
            continue
        group_true = truth[mask]
        group_pred = predicted[mask]
        positives = group_true == positive
        negatives = ~positives
        n_positive = int(positives.sum())
        n_negative = int(negatives.sum())
        hit = group_pred == positive
        n_hit = int(hit.sum())
        tpr_reason = (
            f"group {key!r} has no positive case ({positive_name}) in the held-out test split, "
            "so its true positive rate is undefined"
        )
        fpr_reason = (
            f"group {key!r} has no negative case in the held-out test split, so its false "
            "positive rate is undefined"
        )
        metrics = {
            "accuracy": _measured(float((group_true == group_pred).mean())),
            "selection_rate": _measured(n_hit / n_test),
            "tpr": (
                _measured(float((hit & positives).sum()) / n_positive)
                if n_positive
                else _missing(tpr_reason)
            ),
            "fpr": (
                _measured(float((hit & negatives).sum()) / n_negative)
                if n_negative
                else _missing(fpr_reason)
            ),
        }
        unmeasured = sorted(name for name, entry in metrics.items() if entry["value"] is None)
        rows.append(
            {
                "group": key,
                "n_version": seen,
                "n_total": kept,
                "n_train": kept - n_test,
                "n_test": n_test,
                "n_actual_positive": n_positive,
                "n_actual_negative": n_negative,
                "n_predicted_positive": n_hit,
                "class_counts": _class_counts(group_true, names),
                "measured": not unmeasured,
                "reason": (
                    f"{', '.join(unmeasured)} could not be measured for group {key!r}: "
                    f"{metrics[unmeasured[0]]['reason']}"
                    if unmeasured
                    else None
                ),
                "metrics": metrics,
            }
        )
    return rows


def regression_groups(
    *,
    group_of_row: Sequence[str],
    y_true: Sequence[float],
    y_pred: Sequence[float],
    kept_counts: Mapping[str, int],
    version_counts: Mapping[str, int] | None = None,
) -> list[dict[str, Any]]:
    """MAE (plus RMSE and mean error) for every group, in group order.

    MAE is the metric a regression Fairness Report is about: the average size of
    the miss, per group. ``mean_error`` is the signed average, so a Model that
    over-predicts for one group and under-predicts for another is visible even
    when both groups' MAE is small.
    """
    truth = np.asarray(y_true, dtype="float64")
    predicted = np.asarray(y_pred, dtype="float64")
    _check_lengths(group_of_row, truth, predicted)
    version = dict(version_counts or {})
    rows: list[dict[str, Any]] = []
    for key in _all_groups(group_of_row, kept_counts, version):
        mask = _row_mask(group_of_row, key)
        n_test = int(mask.sum())
        kept = int(kept_counts.get(key, 0))
        seen = int(version.get(key, 0))
        if n_test == 0:
            rows.append(
                _unmeasured_group(
                    key,
                    metrics=REGRESSION_METRICS,
                    kept=kept,
                    seen=seen,
                    reason=_absent_reason(key, kept, seen),
                )
            )
            continue
        residuals = truth[mask] - predicted[mask]
        rows.append(
            {
                "group": key,
                "n_version": seen,
                "n_total": kept,
                "n_train": kept - n_test,
                "n_test": n_test,
                "n_actual_positive": None,
                "n_actual_negative": None,
                "n_predicted_positive": None,
                "class_counts": {},
                "measured": True,
                "reason": None,
                "metrics": {
                    "mae": _measured(float(np.abs(residuals).mean())),
                    "rmse": _measured(float(np.sqrt((residuals**2).mean()))),
                    "mean_error": _measured(float(residuals.mean())),
                },
            }
        )
    return rows


def _class_counts(codes: np.ndarray, names: Sequence[str]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for code in sorted({int(value) for value in codes}):
        label = names[code] if code < len(names) else str(code)
        counts[label] = int((codes == code).sum())
    return counts


# -- the gaps ----------------------------------------------------------------


@dataclass(frozen=True)
class GapSpec:
    """One gap between groups, and whether a rate threshold may judge it."""

    name: str
    label: str
    metric: str
    unit: str
    description: str = ""
    #: The gap in :data:`UNIT_TARGET` whose rate version is judged instead.
    judged_by: str | None = None


#: The gaps a classification Fairness Report produces, most-cited first.
CLASSIFICATION_GAPS: tuple[GapSpec, ...] = (
    GapSpec(
        "demographic_parity",
        "Demographic parity difference",
        "selection_rate",
        UNIT_RATE,
        "the widest difference in the share of each group's rows the Model selects",
    ),
    GapSpec(
        "tpr",
        "True positive rate difference",
        "tpr",
        UNIT_RATE,
        "the widest difference in the share of each group's positive cases the Model catches",
    ),
    GapSpec(
        "fpr",
        "False positive rate difference",
        "fpr",
        UNIT_RATE,
        "the widest difference in the share of each group's negative cases the Model flags",
    ),
    GapSpec(
        "accuracy",
        "Accuracy difference",
        "accuracy",
        UNIT_RATE,
        "the widest difference in accuracy between groups",
    ),
)

#: The gaps a regression Fairness Report produces. The absolute MAE gap is in the
#: Target's own units, so only the relative one is a rate and only it is judged.
REGRESSION_GAPS: tuple[GapSpec, ...] = (
    GapSpec(
        "mae",
        "MAE difference",
        "mae",
        UNIT_TARGET,
        "the widest difference in mean absolute error, in the Target's own units",
        judged_by="mae_relative",
    ),
    GapSpec(
        "mae_relative",
        "MAE difference (relative)",
        "mae",
        UNIT_RATE,
        "the MAE difference as a share of the Model's overall MAE, so it can be compared "
        "with the rate threshold",
    ),
)


def gaps_for(task_type: str) -> tuple[GapSpec, ...]:
    """The gap specs a Task Type produces."""
    if task_type == CLASSIFICATION:
        return CLASSIFICATION_GAPS
    if task_type == REGRESSION:
        return REGRESSION_GAPS
    raise FairnessError(
        f"a Fairness Report needs a classification or regression Target, got {task_type!r}"
    )


def _spread(
    groups: Sequence[Mapping[str, Any]], metric: str
) -> tuple[list[tuple[str, float]], list[str]]:
    """The groups that have a value for ``metric``, and those that do not."""
    have: list[tuple[str, float]] = []
    missing: list[str] = []
    for group in groups:
        value = (group.get("metrics") or {}).get(metric, {}).get("value")
        if value is None:
            missing.append(str(group["group"]))
        else:
            have.append((str(group["group"]), float(value)))
    return have, missing


def metric_gap(
    spec: GapSpec,
    groups: Sequence[Mapping[str, Any]],
    *,
    threshold: float,
    overall: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """One gap: the widest difference between the groups that have a value.

    A group whose metric is ``null`` is **skipped, never imputed**: the gap says
    how many groups it used and names the ones it could not use, so a two-group
    gap is never mistaken for a three-group one. Fewer than two usable groups and
    the gap is ``null`` with the reason.
    """
    have, missing = _spread(groups, spec.metric)
    base: dict[str, Any] = {
        "name": spec.name,
        "label": spec.label,
        "metric": spec.metric,
        "unit": spec.unit,
        "description": spec.description,
        "threshold": float(threshold),
        "groups_total": len(groups),
        "groups_compared": len(have),
        "groups_unmeasured": missing,
    }
    if len(have) < 2:
        why = (
            f"this Dataset Version holds a single group ({have[0][0]}), so there is nothing to compare"
            if len(have) == 1
            else f"no group has a {spec.metric} on the held-out test split"
        )
        return {
            **base,
            "value": None,
            "highest_group": None,
            "lowest_group": None,
            "highest_value": None,
            "lowest_value": None,
            "comparable": spec.unit == UNIT_RATE,
            "exceeds": None,
            "reason": why,
        }
    # Sort once, so a tie between two groups resolves the same way every time.
    ordered = sorted(have, key=lambda item: (item[1], item[0]))
    lowest, highest = ordered[0], ordered[-1]
    value = highest[1] - lowest[1]
    entry: dict[str, Any] = {
        **base,
        "value": round(float(value), _ROUND),
        "highest_group": highest[0],
        "lowest_group": lowest[0],
        "highest_value": round(float(highest[1]), _ROUND),
        "lowest_value": round(float(lowest[1]), _ROUND),
    }
    if spec.name == "mae_relative":
        baseline = float((overall or {}).get("value") or 0.0)
        if not math.isfinite(baseline) or abs(baseline) < _TINY:
            return {
                **entry,
                "value": None,
                "comparable": True,
                "exceeds": None,
                "reason": (
                    "the Model's overall MAE is 0 on the held-out test split, so the MAE gap as "
                    "a share of it is undefined"
                ),
            }
        relative = float(value) / abs(baseline)
        return {
            **entry,
            "value": round(relative, _ROUND),
            "overall_mae": round(abs(baseline), _ROUND),
            "comparable": True,
            "exceeds": relative > float(threshold),
            "reason": None,
        }
    if spec.unit != UNIT_RATE:
        return {
            **entry,
            "comparable": False,
            "exceeds": None,
            "reason": (
                f"this gap is in the Target's own units, not a rate, so it is not compared with "
                f"the threshold of {threshold}; {spec.judged_by} carries the comparable version"
            ),
        }
    return {**entry, "comparable": True, "exceeds": float(value) > float(threshold), "reason": None}


def measured_gaps(gaps: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """The gaps that have a value, widest first (ties by name).

    Includes the ones a rate threshold cannot judge — the absolute MAE gap, say —
    because "the widest gap I can see" is a fact about the Model regardless of
    whether a threshold is allowed to call it a finding.
    """
    usable = [dict(gap) for gap in gaps if gap.get("value") is not None]
    return sorted(usable, key=lambda gap: (-float(gap["value"]), str(gap["name"])))


def comparable_gaps(gaps: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """The measured gaps a threshold can judge, widest first."""
    usable = [
        dict(gap)
        for gap in gaps
        if gap.get("comparable")
        and gap.get("value") is not None
        and gap.get("exceeds") is not None
    ]
    return sorted(usable, key=lambda gap: (-float(gap["value"]), str(gap["name"])))


def exceeding_gaps(gaps: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """The comparable gaps that clear the threshold, widest first."""
    return [gap for gap in comparable_gaps(gaps) if gap.get("exceeds") is True]


# -- the report --------------------------------------------------------------


@dataclass(frozen=True)
class ReportSource:
    """Who the report is about: the run, the Model, the attribute and the threshold."""

    training_run_id: str
    version_id: str
    target: str
    task_type: str
    model: str
    model_label: str
    sensitive_attribute: str
    #: The attribute the Training Run declared, if it declared one.
    declared_sensitive_attribute: str | None = None
    #: Whether the Model was denied this column (#31 excludes it by default).
    excluded_from_features: bool = True
    #: The label the caller asked to treat as positive, verbatim; ``None`` = the default.
    positive_label: str | None = None
    threshold: float = DEFAULT_GAP_THRESHOLD
    threshold_source: str = "default"
    project_id: str | None = None
    train_rows: int = 0
    test_rows: int = 0
    refit: bool = True
    hyperparameters: dict[str, Any] = field(default_factory=dict)
    seed: Any = None


def _overall_classification(
    y_true: np.ndarray, y_pred: np.ndarray, positive: int
) -> dict[str, Any]:
    if not len(y_true):
        why = "the held-out test split is empty"
        return {
            "accuracy": _missing(why),
            "selection_rate": _missing(why),
            "n_test": 0,
        }
    return {
        "accuracy": _measured(float((y_true == y_pred).mean())),
        "selection_rate": _measured(float((y_pred == positive).mean())),
        "n_test": len(y_true),
    }


def _overall_regression(residuals: np.ndarray) -> dict[str, Any]:
    if not len(residuals):
        why = "the held-out test split is empty"
        return {
            "mae": _missing(why),
            "rmse": _missing(why),
            "mean_error": _missing(why),
            "n_test": 0,
        }
    return {
        "mae": _measured(float(np.abs(residuals).mean())),
        "rmse": _measured(float(np.sqrt((residuals**2).mean()))),
        "mean_error": _measured(float(residuals.mean())),
        "n_test": len(residuals),
    }


def build_report(
    source: ReportSource,
    *,
    group_of_row: Sequence[str],
    y_true: Sequence[Any],
    y_pred: Sequence[Any],
    kept_counts: Mapping[str, int],
    version_counts: Mapping[str, int] | None = None,
    classes: Sequence[str] | None = None,
    positive: int | None = None,
) -> dict[str, Any]:
    """The whole Fairness Report, as a plain JSON-able dict.

    Pure: given the group column, the held-out truth and the Model's held-out
    predictions, it computes everything — no store, no estimator, no sklearn — so
    it is testable on a hand-built fixture.
    """
    task_type = source.task_type
    if task_type not in (CLASSIFICATION, REGRESSION):
        raise FairnessError(
            f"a Fairness Report needs a classification or regression Target, got {task_type!r}"
        )
    names = [str(c) for c in (classes or [])]
    positive_class: str | None = None

    if task_type == CLASSIFICATION:
        code = positive
        if code is None:
            code, positive_class = positive_class_code(names, source.positive_label)
        else:
            positive_class = names[code] if 0 <= code < len(names) else str(code)
        truth = np.asarray(y_true, dtype="int64")
        predicted = np.asarray(y_pred, dtype="int64")
        groups = classification_groups(
            group_of_row=group_of_row,
            y_true=truth,
            y_pred=predicted,
            positive=code,
            kept_counts=kept_counts,
            version_counts=version_counts,
            classes=names,
        )
        overall = _overall_classification(truth, predicted, code)
        gaps = [
            metric_gap(spec, groups, threshold=source.threshold)
            for spec in gaps_for(CLASSIFICATION)
        ]
    else:
        truth = np.asarray(y_true, dtype="float64")
        predicted = np.asarray(y_pred, dtype="float64")
        groups = regression_groups(
            group_of_row=group_of_row,
            y_true=truth,
            y_pred=predicted,
            kept_counts=kept_counts,
            version_counts=version_counts,
        )
        overall = _overall_regression(truth - predicted)
        gaps = [
            metric_gap(spec, groups, threshold=source.threshold, overall=overall.get("mae") or {})
            for spec in gaps_for(REGRESSION)
        ]

    ranked = comparable_gaps(gaps)
    over = [gap for gap in ranked if gap["exceeds"] is True]
    widest = measured_gaps(gaps)
    unmeasured = [group for group in groups if not group.get("measured")]
    return {
        "training_run_id": source.training_run_id,
        "project_id": source.project_id,
        "version_id": source.version_id,
        "target": source.target,
        "task_type": task_type,
        "model": source.model,
        "model_label": source.model_label,
        "hyperparameters": dict(source.hyperparameters),
        "seed": source.seed,
        "refit": bool(source.refit),
        "sensitive_attribute": source.sensitive_attribute,
        "declared_sensitive_attribute": source.declared_sensitive_attribute,
        "sensitive_attribute_excluded_from_features": bool(source.excluded_from_features),
        "positive_label": source.positive_label,
        "positive_class": positive_class,
        "classes": names,
        "threshold": float(source.threshold),
        "threshold_source": source.threshold_source,
        "split": {
            "train_rows": int(source.train_rows),
            "test_rows": int(source.test_rows),
            "reused_stored_split": True,
        },
        "overall": overall,
        "n_groups": len(groups),
        "n_groups_measured": sum(1 for group in groups if group.get("measured")),
        "groups": groups,
        "unmeasured_groups": [
            {"group": group["group"], "reason": group["reason"], "n_test": group["n_test"]}
            for group in unmeasured
        ],
        "gaps": gaps,
        "gaps_exceeding_threshold": [gap["name"] for gap in over],
        #: The widest gap of any kind — the one to read first, whether or not a
        #: rate threshold is allowed to call it a finding.
        "largest_gap": widest[0] if widest else None,
        #: The widest gap a threshold *can* judge, even when nothing exceeded it.
        "largest_comparable_gap": ranked[0] if ranked else None,
        "gaps_within_threshold": [gap["name"] for gap in ranked if gap["exceeds"] is False],
        "gaps_not_compared": [
            gap["name"] for gap in gaps if gap.get("value") is not None and not gap.get("comparable")
        ],
        "notes": _notes(source, groups=groups, unmeasured=unmeasured),
    }


def _notes(
    source: ReportSource,
    *,
    groups: Sequence[Mapping[str, Any]],
    unmeasured: Sequence[Mapping[str, Any]],
) -> list[str]:
    """The honest caveats, in the order they matter to a reader."""
    notes: list[str] = []
    attribute = source.sensitive_attribute
    if not source.excluded_from_features:
        notes.append(
            f"Sensitive Attribute {attribute!r} was a feature of this Model, so the Model was "
            "allowed to see it. These gaps measure what the Model did with the attribute, not "
            "how it treats people it cannot identify."
        )
    if source.declared_sensitive_attribute not in (None, attribute):
        notes.append(
            f"This Training Run declared {source.declared_sensitive_attribute!r} as its Sensitive "
            f"Attribute; the report was asked for {attribute!r} instead."
        )
    if any(group["group"] == MISSING_GROUP for group in groups):
        notes.append(
            f"Some rows have no value for {attribute!r}; they are reported as the "
            f"{MISSING_GROUP!r} group rather than dropped."
        )
    if unmeasured:
        names = ", ".join(str(group["group"]) for group in unmeasured)
        notes.append(
            f"{len(unmeasured)} group(s) produced no complete metric and are left out of the gaps: "
            f"{names}. Every gap says how many groups it used."
        )
    if sum(1 for group in groups if group["n_test"] > 0) < 2:
        notes.append(
            "fewer than 2 groups have held-out rows, so no gap between groups can be measured."
        )
    return notes


# -- rebuilding the run ------------------------------------------------------


def load_run_context(store: Any, job: Any) -> RunContext:
    """Rebuild the run's rows, split and preprocessing — and prove they are its own.

    Thin wrapper over :func:`.run_rows.resolve_run`, which owns the rebuild and
    the check that the stored split still reproduces. Only the refusal is
    re-voiced here, so a report can say so in the report's own terms.
    """
    try:
        return resolve_run(store, job)
    except ValueError as exc:
        raise FairnessError(str(exc)) from exc


def leaderboard_entry(run: Mapping[str, Any], model: str | None = None) -> dict[str, Any]:
    """The leaderboard row to report on: the one named, else the top-ranked Model."""
    board = [dict(entry) for entry in (run.get("leaderboard") or [])]
    if not board:
        raise FairnessError(
            "this Training Run has an empty leaderboard, so there is no Model to report on"
        )
    if model is not None:
        for entry in board:
            if entry.get("model") == model:
                if entry.get("status") != "ok":
                    raise FairnessError(
                        f"Model {model!r} failed to fit in this Training Run "
                        f"({entry.get('error') or 'no error recorded'}), so it has no predictions "
                        "to compare groups on"
                    )
                return entry
        known = ", ".join(str(entry.get("model")) for entry in board)
        raise FairnessError(
            f"Model {model!r} is not on this Training Run's leaderboard; it has: {known}"
        )
    ranked = [
        entry for entry in board if entry.get("status") == "ok" and entry.get("rank") is not None
    ]
    ranked.sort(key=lambda entry: int(entry["rank"]))
    if ranked:
        return ranked[0]
    fitted = [entry for entry in board if entry.get("status") == "ok"]
    if fitted:
        return fitted[0]
    raise FairnessError(
        "every Model in this Training Run failed to fit, so there is nothing to report on"
    )


def refit_model(context: RunContext, entry: Mapping[str, Any]) -> Any:
    """Refit the Model the run fitted, from the run's own data and hyperparameters.

    Deterministic: the same matrix, the same Target codes, and the same
    hyperparameters (seed included) that the leaderboard row records. Returns the
    :class:`~dat_distiller.training.trainers.FittedModel`, so the caller predicts
    exactly as :func:`train._score` would.
    """
    name = str(entry.get("model"))
    if name not in MODEL_SPECS:
        raise FairnessError(f"unknown Model {name!r} on this Training Run's leaderboard")
    spec = get_spec(name)
    if not is_installed(spec.extra):
        raise FairnessError(
            f"Model {spec.label} needs the optional extra {spec.extra!r}, which this install does "
            f"not have; install it with `uv sync --extra {spec.extra}` to compute a Fairness Report"
        )
    matrix = context.train_matrix()
    return fit_model(
        spec,
        context.setup.task_type,
        matrix.X,
        matrix.y,
        hyperparameters=entry.get("hyperparameters") or {},
        classes=matrix.classes,
    )


def predict_held_out(context: RunContext, fitted: Any) -> tuple[np.ndarray, np.ndarray]:
    """The held-out truth and the Model's predictions, as class codes."""
    test_frame = context.test_frame
    if context.setup.task_type == CLASSIFICATION:
        return (
            encode_with_classes(test_frame[context.setup.target].to_numpy(), context.classes),
            np.asarray(fitted.predict(context.test_matrix()), dtype="int64"),
        )
    try:
        truth = encode_regression(test_frame[context.setup.target].to_numpy())
    except ModelError as exc:
        raise FairnessError(str(exc)) from exc
    return truth, np.asarray(fitted.predict(context.test_matrix()), dtype="float64")


# -- the on-demand report ----------------------------------------------------


def fairness_report(
    *,
    store: Any,
    job: Any,
    sensitive_attribute: str,
    model: str | None = None,
    positive_label: Any = None,
    threshold: Any = None,
    threshold_source: str = "settings",
) -> dict[str, Any]:
    """Compute the Fairness Report for one Model, split by one Sensitive Attribute.

    The whole on-demand path: the run's stored pipeline is reused, the Model is
    refitted from it, the held-out rows are predicted once, and the report is
    built. Only held-out rows produce a group metric — a fairness finding is a
    finding about the Model's behaviour, which is what the held-out split
    measures.
    """
    if not str(sensitive_attribute or "").strip():
        raise FairnessError("name a Sensitive Attribute to build a Fairness Report")
    attribute = str(sensitive_attribute)
    context = load_run_context(store, job)
    setup = context.setup
    if attribute not in context.frame.columns:
        raise FairnessError(
            f"unknown Sensitive Attribute {attribute!r}; this Dataset Version has "
            f"{[str(c) for c in context.frame.columns]}"
        )
    if attribute == setup.target:
        raise FairnessError("the Sensitive Attribute cannot also be the Target")
    if not context.test_indices:
        raise FairnessError(
            "this Training Run held out no rows, so a Fairness Report has nothing to measure"
        )

    entry = leaderboard_entry(context.run, model)
    fitted = refit_model(context, entry)
    truth, predicted = predict_held_out(context, fitted)
    source = ReportSource(
        training_run_id=str(context.run.get("training_run_id") or job.id),
        version_id=str(setup.version_id),
        target=setup.target,
        task_type=setup.task_type,
        model=str(entry.get("model")),
        model_label=str(entry.get("label") or get_spec(str(entry["model"])).label),
        sensitive_attribute=attribute,
        declared_sensitive_attribute=setup.sensitive_attribute,
        excluded_from_features=attribute not in set(setup.feature_columns),
        positive_label=None if positive_label is None else str(positive_label),
        threshold=resolve_gap_threshold(threshold),
        threshold_source=threshold_source,
        project_id=context.run.get("project_id") or job.project_id,
        train_rows=len(context.train_indices),
        test_rows=len(context.test_indices),
        hyperparameters=dict(entry.get("hyperparameters") or {}),
        seed=context.run.get("seed"),
    )
    return build_report(
        source,
        group_of_row=group_keys(context.test_frame[attribute].to_numpy()),
        y_true=truth,
        y_pred=predicted,
        kept_counts=group_sizes(context.frame[attribute].to_numpy()),
        version_counts=group_sizes(context.source_frame[attribute].to_numpy()),
        classes=context.classes,
    )


# -- the Checks --------------------------------------------------------------


def _gap_message(gap: Mapping[str, Any], context: Mapping[str, Any], *, largest: bool) -> str:
    """One gap, in words, naming the groups and the numbers behind it."""
    message = (
        f"Model {context['model']!r} by {context['attribute']!r}: {gap['label'].lower()} "
        f"{gap['value']:.3f} exceeds the threshold of {gap['threshold']} — "
        f"{gap['highest_group']} at {gap['highest_value']:.3f} against {gap['lowest_group']} at "
        f"{gap['lowest_value']:.3f} (compared across {gap['groups_compared']} of "
        f"{gap['groups_total']} group(s))"
    )
    if largest:
        return f"{message}; the widest of {context['n_exceeding']} gap(s) over the threshold"
    if context.get("wider"):
        return f"{message}; {context['wider']} is wider"
    return message


def _registered_keys(checks_store: Any, run_id: str) -> set[tuple[str, str, str]]:
    """The (Model, Sensitive Attribute, gap) triples already checked for a run."""
    keys: set[tuple[str, str, str]] = set()
    for check in checks_store.list_for_subject("training_run", run_id):
        if check.kind not in (FAIRNESS_GAP_CHECK_KIND, FAIRNESS_USED_CHECK_KIND):
            continue
        keys.add(
            (
                str(check.details.get("model")),
                str(check.details.get("sensitive_attribute")),
                str(check.details.get("gap") or check.kind),
            )
        )
    return keys


def register_fairness_checks(
    checks_store: Any,
    report: Mapping[str, Any],
    *,
    step: str = FAIRNESS_STEP,
) -> list[Any]:
    """Register the report's Checks against the Training Run.

    One **warning** Check per gap over the threshold, each naming its own groups
    and numbers; the widest one says it is the widest. Deduplicated per
    ``(Model, Sensitive Attribute, gap)`` — ``CheckStore.has_check`` is
    subject-wide and cannot express that, so the run's existing Checks are read
    and keyed here. A repeated request with the same threshold therefore adds
    nothing, while one with a *different* threshold still reports.

    The **info** Check for a Sensitive Attribute the Model was allowed to use is
    registered the same way, once per (Model, attribute): a Fairness Report on a
    column the Model could see is a different finding, and it belongs in the
    Training Run's Checks.
    """
    run_id = str(report.get("training_run_id") or "")
    over = exceeding_gaps(report.get("gaps") or [])
    widest = measured_gaps(report.get("gaps") or [])
    # The Check's "this is the widest" marker is about the *flagged* gaps: the
    # Check itself exists to say which of the gaps over the threshold is the
    # finding, so it must not be stolen by a gap no threshold could judge.
    largest = over[0]["name"] if over else None
    seen = _registered_keys(checks_store, run_id)
    details_base = {
        "training_run_id": report.get("training_run_id"),
        "version_id": report.get("version_id"),
        "model": report.get("model"),
        "model_label": report.get("model_label"),
        "target": report.get("target"),
        "task_type": report.get("task_type"),
        "sensitive_attribute": report.get("sensitive_attribute"),
        "declared_sensitive_attribute": report.get("declared_sensitive_attribute"),
        "sensitive_attribute_excluded_from_features": report.get(
            "sensitive_attribute_excluded_from_features"
        ),
        "positive_class": report.get("positive_class"),
        "test_rows": (report.get("split") or {}).get("test_rows"),
        "n_groups": report.get("n_groups"),
    }
    context: dict[str, Any] = {
        "model": report.get("model_label") or report.get("model"),
        "attribute": report.get("sensitive_attribute"),
        "n_exceeding": len(over),
    }
    registered: list[Any] = []
    for gap in over:
        key = (str(report.get("model")), str(report.get("sensitive_attribute")), str(gap["name"]))
        if key in seen:
            continue
        context["wider"] = ", ".join(
            other["name"] for other in over if float(other["value"]) > float(gap["value"])
        )
        registered.append(
            checks_store.register(
                kind=FAIRNESS_GAP_CHECK_KIND,
                severity="warning",
                message=_gap_message(gap, context, largest=gap["name"] == largest),
                subject_type="training_run",
                subject_id=run_id,
                details={
                    **details_base,
                    "gap": gap["name"],
                    "label": gap["label"],
                    "metric": gap["metric"],
                    "value": gap["value"],
                    "threshold": gap["threshold"],
                    "highest_group": gap["highest_group"],
                    "highest_value": gap["highest_value"],
                    "lowest_group": gap["lowest_group"],
                    "lowest_value": gap["lowest_value"],
                    "groups_compared": gap["groups_compared"],
                    "groups_total": gap["groups_total"],
                    "groups_unmeasured": gap["groups_unmeasured"],
                    "largest_gap": largest,
                    "largest_gap_value": widest[0]["value"] if widest else None,
                    "gaps_exceeding_threshold": [entry["name"] for entry in over],
                },
                step=step,
            )
        )
        seen.add(key)

    if report.get("sensitive_attribute_excluded_from_features") is False:
        key = (
            str(report.get("model")),
            str(report.get("sensitive_attribute")),
            FAIRNESS_USED_CHECK_KIND,
        )
        if key not in seen:
            registered.append(
                checks_store.register(
                    kind=FAIRNESS_USED_CHECK_KIND,
                    severity="info",
                    message=(
                        f"A Fairness Report was computed on {report.get('sensitive_attribute')!r}, "
                        "which was a feature of this Model: the gaps measure how the Model used "
                        "the attribute, not how it treats people it cannot identify"
                    ),
                    subject_type="training_run",
                    subject_id=run_id,
                    details={**details_base, "note": "sensitive attribute was a feature"},
                    step=step,
                )
            )
            seen.add(key)
    return registered


__all__ = [
    "CLASSIFICATION_GAPS",
    "CLASSIFICATION_METRICS",
    "DEFAULT_GAP_THRESHOLD",
    "FAIRNESS_GAP_CHECK_KIND",
    "FAIRNESS_STEP",
    "FAIRNESS_USED_CHECK_KIND",
    "MISSING_GROUP",
    "REGRESSION_GAPS",
    "REGRESSION_METRICS",
    "UNIT_RATE",
    "UNIT_TARGET",
    "FairnessError",
    "GapSpec",
    "ReportSource",
    "RunContext",
    "build_report",
    "classification_groups",
    "comparable_gaps",
    "encode_with_classes",
    "exceeding_gaps",
    "fairness_report",
    "gaps_for",
    "group_keys",
    "group_sizes",
    "leaderboard_entry",
    "load_run_context",
    "measured_gaps",
    "metric_gap",
    "positive_class_code",
    "predict_held_out",
    "refit_model",
    "register_fairness_checks",
    "regression_groups",
    "resolve_gap_threshold",
]
