"""Checks for a Training Run: what the setup found risky before any fitting.

Five Checks, all against ``subject_type="training_run"``:

- ``class_imbalance`` (warning) — a Target class is a rounding error away
- ``train_test_duplicates`` (warning) — the held-out split shares rows with
  training, so its metrics flatter the Model
- ``target_leakage`` (warning) — one feature predicts the Target on its own
- ``unreviewed_labels`` (warning, or info once they are excluded) — low
  confidence Jev labels the user chose to train on
- ``label_sibling_exclusion`` (info) — the leakage guard removed a Label
  Column's confidence/probability siblings

The evaluation is separated from the registration so each condition is testable
on its own and the whole report is a plain dict for the job row.
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Sequence
from typing import Any

import numpy as np
import pandas as pd

#: A class smaller than this share of the majority is flagged.
IMBALANCE_MIN_FRACTION = 0.1
#: A single feature that predicts the Target this well is flagged.
LEAKAGE_MIN_SCORE = 0.95
#: Above this many Target classes the single-feature screen is not meaningful.
MAX_LEAKAGE_CLASSES = 50
#: Separator between the cell values of one row's key.
_KEY_SEPARATOR = "\x1f"
_NA_KEY = "\x00"


def value_key(value: Any) -> str:
    """A stable string for one Target value, safe for a JSON report and for
    counting the same class the same way everywhere in the Train step."""
    if value is None or value is pd.NA:
        return "missing"
    if isinstance(value, (bool, np.bool_)):
        return f"bool:{bool(value)}"
    if isinstance(value, (int, np.integer)):
        return f"int:{int(value)}"
    if isinstance(value, (float, np.floating)):
        if math.isnan(float(value)):
            return "missing"
        return f"num:{float(value):.6g}"
    return f"str:{value}"


def _finding(raised: bool, severity: str | None, message: str, details: dict[str, Any]) -> dict[str, Any]:
    return {
        "raised": bool(raised),
        "severity": severity if raised else None,
        "message": message,
        "details": details,
    }


def _skipped(reason: str) -> dict[str, Any]:
    return {"raised": False, "severity": None, "message": "", "details": {"skipped": reason}}


# -- class imbalance --------------------------------------------------------


def class_balance(values: Iterable[Any]) -> dict[str, Any]:
    """Per-class counts, shares, the smallest share and the imbalance ratio.

    A Target with no values has no minority class and no imbalance, so the two
    derived figures come back null with a reason rather than as 0.0 and 1.0.
    A fabricated "perfectly balanced" reads as a clean result and would pass a
    gate, when the truth is that there is nothing to measure.
    """
    counts: dict[str, int] = {}
    for value in values:
        key = value_key(value)
        counts[key] = counts.get(key, 0) + 1
    total = sum(counts.values())
    if total == 0:
        return {
            "total": 0,
            "classes": 0,
            "counts": {},
            "fractions": {},
            "min_fraction": None,
            "imbalance_ratio": None,
            "reason": "the Target has no values, so there is no class to measure",
        }
    fractions = {key: round(count / total, 4) for key, count in counts.items()}
    largest = max(counts.values())
    smallest = min(counts.values())
    return {
        "total": int(total),
        "classes": len(counts),
        "counts": counts,
        "fractions": fractions,
        "min_fraction": min(fractions.values()),
        "imbalance_ratio": round(largest / smallest, 4) if smallest else float("inf"),
    }


def class_imbalance_check(setup: Any, values: Sequence[Any]) -> dict[str, Any]:
    if setup.task_type != "classification":
        return _finding(
            False, None, "", {"task_type": setup.task_type, "reason": "not a classification Target"}
        )
    balance = class_balance(values)
    if not balance["classes"]:
        return _skipped("no Target values")
    smallest = min(balance["counts"], key=lambda key: balance["counts"][key])
    imbalance = balance["min_fraction"] < IMBALANCE_MIN_FRACTION
    message = (
        f"Target {setup.target!r} is imbalanced: {balance['counts']} "
        f"(smallest class {smallest} at {balance['min_fraction']:.0%})"
        if imbalance
        else ""
    )
    return _finding(
        imbalance,
        "warning",
        message,
        {
            "target": setup.target,
            "task_type": setup.task_type,
            "min_fraction": balance["min_fraction"],
            "imbalance_ratio": balance["imbalance_ratio"],
            "counts": balance["counts"],
            "threshold": IMBALANCE_MIN_FRACTION,
        },
    )


# -- train/test duplicates --------------------------------------------------


def _row_keys(frame: pd.DataFrame) -> list[str]:
    """One NA-aware string key per row over the given columns."""
    keys: list[str] = []
    for row in frame.to_numpy(dtype=object):
        cells = [
            _NA_KEY if (cell is None or cell is pd.NA) else str(cell)
            for cell in row
        ]
        keys.append(_KEY_SEPARATOR.join(cells))
    return keys


def duplicate_rows(
    train_frame: pd.DataFrame, test_frame: pd.DataFrame, columns: Sequence[str]
) -> dict[str, Any]:
    """Rows that appear on both sides of the split — the test split is not then held out."""
    train_keys = _row_keys(train_frame[list(columns)])
    test_keys = _row_keys(test_frame[list(columns)])
    counts: dict[str, int] = {}
    for key in train_keys:
        counts[key] = counts.get(key, 0) + 1
    overlapping = {key for key in test_keys if key in counts}
    duplicated = int(sum(counts[key] for key in overlapping))
    fraction = round(duplicated / len(test_keys), 4) if test_keys else 0.0
    examples = [
        key.split(_KEY_SEPARATOR)
        for key in sorted(overlapping)[:3]
    ]
    return {
        "count": duplicated,
        "test_rows": len(test_keys),
        "fraction": fraction,
        "examples": examples,
        "columns": list(columns),
    }


def train_test_duplicates_check(setup: Any, train_frame: pd.DataFrame, test_frame: pd.DataFrame) -> dict[str, Any]:
    found = duplicate_rows(
        train_frame, test_frame, [*setup.feature_columns, setup.target]
    )
    raised = found["count"] > 0
    message = (
        f"{found['count']} row(s) are identical in the training and test splits "
        f"({found['fraction']:.0%} of the test split); its metrics are optimistic"
        if raised
        else ""
    )
    return _finding(raised, "warning", message, {**found, "target": setup.target})


# -- target leakage ---------------------------------------------------------


def _one_hot(values: Sequence[Any]) -> tuple[np.ndarray, int]:
    keys = [value_key(value) for value in values]
    classes = sorted(set(keys))
    lookup = {key: position for position, key in enumerate(classes)}
    matrix = np.zeros((len(keys), len(classes)), dtype="float64")
    for row, key in enumerate(keys):
        matrix[row, lookup[key]] = 1.0
    return matrix, len(classes)


def _stump_accuracy(feature: np.ndarray, one_hot: np.ndarray) -> float:
    """Best accuracy a single threshold on one feature can reach.

    Sort the rows by the feature, then take the most accurate one-threshold cut:
    each side predicts its own majority class, so the errors on a side are the
    rows that are *not* its majority. Cheap, parameter-free, and exactly the
    shape of "this column basically *is* the Target".
    """
    rows = len(feature)
    if rows == 0 or one_hot.shape[1] == 0:
        return 0.0
    order = np.argsort(feature, kind="mergesort")
    cumulative = np.cumsum(one_hot[order], axis=0)
    left = np.vstack([np.zeros((1, one_hot.shape[1]), dtype="float64"), cumulative])
    right = cumulative[-1] - left
    left_sizes = np.arange(rows + 1, dtype="float64")
    right_sizes = rows - left_sizes
    errors = (left_sizes - left.max(axis=1)) + (right_sizes - right.max(axis=1))
    return float(1.0 - errors.min() / rows)


def _pearson(left: np.ndarray, right: np.ndarray) -> float:
    left_centred = left - left.mean()
    right_centred = right - right.mean()
    denominator = float(np.linalg.norm(left_centred) * np.linalg.norm(right_centred))
    if denominator == 0:
        return 0.0
    return float(np.clip(left_centred @ right_centred / denominator, -1.0, 1.0))


def single_feature_scores(
    matrix: np.ndarray,
    values: Sequence[Any],
    feature_names: Sequence[str],
    *,
    task_type: str,
) -> dict[str, float]:
    """How well each single column predicts the Target, on its own."""
    scores: dict[str, float] = {}
    if matrix.size == 0 or len(values) == 0:
        return scores
    if task_type == "classification":
        one_hot, classes = _one_hot(values)
        if classes > MAX_LEAKAGE_CLASSES:
            return scores
        for position, name in enumerate(feature_names[: matrix.shape[1]]):
            scores[name] = _stump_accuracy(matrix[:, position], one_hot)
        return scores
    numeric = np.asarray([float(_to_float(value)) for value in values], dtype="float64")
    for position, name in enumerate(feature_names[: matrix.shape[1]]):
        scores[name] = _pearson(matrix[:, position].astype("float64"), numeric) ** 2
    return scores


def _to_float(value: Any) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return float("nan")
    return result


def detect_target_leakage(
    matrix: np.ndarray,
    values: Sequence[Any],
    feature_names: Sequence[str],
    *,
    task_type: str,
    sources: dict[str, str] | None = None,
    min_score: float = LEAKAGE_MIN_SCORE,
    max_classes: int = MAX_LEAKAGE_CLASSES,
) -> dict[str, Any]:
    """Features that predict the Target on their own — a warning-worthy leak.

    Classification is scored by the best single-threshold accuracy (AUC in one
    dimension); regression by the R² of a one-feature straight line. A derived
    column is reported under the source column it came from.
    """
    if matrix is None or matrix.size == 0:
        return {"count": 0, "threshold": min_score, "suspects": [], "skipped": "no matrix"}
    if task_type == "classification":
        _, classes = _one_hot(values)
        if classes > max_classes:
            return {
                "count": 0,
                "threshold": min_score,
                "suspects": [],
                "skipped": f"more than {max_classes} Target classes",
            }
    scores = single_feature_scores(matrix, values, feature_names, task_type=task_type)
    metric = "stump_accuracy" if task_type == "classification" else "r_squared"
    suspects = [
        {
            "feature": name,
            "column": (sources or {}).get(name, name),
            "score": round(float(score), 4),
            "metric": metric,
        }
        for name, score in sorted(scores.items(), key=lambda item: -item[1])
        if score >= min_score
    ]
    return {"count": len(suspects), "threshold": min_score, "metric": metric, "suspects": suspects}


def target_leakage_check(
    setup: Any,
    matrix: np.ndarray | None,
    values: Sequence[Any] | None,
    sources: dict[str, str] | None = None,
    feature_names: Sequence[str] | None = None,
) -> dict[str, Any]:
    if matrix is None or values is None:
        return _skipped("the training matrix was not built")
    found = detect_target_leakage(
        matrix,
        values,
        feature_names or [f"f{index}" for index in range(matrix.shape[1])],
        task_type=setup.task_type,
        sources=sources,
    )
    raised = found["count"] > 0
    names = ", ".join(sorted({suspect["column"] for suspect in found["suspects"]}))
    message = (
        f"Feature(s) {names} predict Target {setup.target!r} on their own "
        f"(score >= {found['threshold']}) and leak the answer"
        if raised
        else ""
    )
    return _finding(raised, "warning", message, {"target": setup.target, **found})


# -- unreviewed labels ------------------------------------------------------


def unreviewed_labels_check(setup: Any) -> dict[str, Any]:
    details = {
        "target": setup.target,
        "family": setup.label_family,
        "confidence_column": setup.confidence_column,
        "threshold": setup.review_threshold,
        "unreviewed_rows": int(setup.unreviewed_rows),
        "excluded_rows": int(setup.dropped_unreviewed),
        "reviewed_rows": int(setup.reviewed_rows),
    }
    if setup.unreviewed_rows > 0:
        return _finding(
            True,
            "warning",
            f"{setup.unreviewed_rows} row(s) keep a Jev label below the review threshold "
            f"({setup.review_threshold}) without human review",
            details,
        )
    if setup.dropped_unreviewed > 0:
        return _finding(
            True,
            "info",
            f"{setup.dropped_unreviewed} low-confidence unreviewed row(s) were excluded",
            details,
        )
    return _finding(False, None, "", details)


# -- Label Column siblings (info) -------------------------------------------


def label_sibling_exclusion_check(setup: Any) -> dict[str, Any]:
    excluded = sorted(
        column for column, reason in setup.excluded_columns.items() if reason == "target_sibling"
    )
    details = {
        "target": setup.target,
        "family": setup.label_family,
        "excluded": excluded,
        "count": len(excluded),
    }
    if not excluded:
        return _finding(False, None, "", details)
    return _finding(
        True,
        "info",
        f"Excluded {len(excluded)} sibling column(s) of Target {setup.target!r} "
        f"so the Label Column's own answer cannot leak in: {excluded}",
        details,
    )


# -- the whole report -------------------------------------------------------

CHECK_KINDS = (
    "class_imbalance",
    "train_test_duplicates",
    "target_leakage",
    "unreviewed_labels",
    "label_sibling_exclusion",
)


def evaluate_training_checks(
    *,
    setup: Any,
    train_frame: pd.DataFrame,
    test_frame: pd.DataFrame,
    matrix: np.ndarray | None = None,
    values: Sequence[Any] | None = None,
    sources: dict[str, str] | None = None,
    feature_names: Sequence[str] | None = None,
) -> dict[str, Any]:
    """Run every Check for a Training Run and return a plain, JSON-able report."""
    target_values = train_frame[setup.target].to_numpy() if values is None else values
    return {
        "class_imbalance": class_imbalance_check(setup, target_values),
        "train_test_duplicates": train_test_duplicates_check(setup, train_frame, test_frame),
        "target_leakage": target_leakage_check(setup, matrix, target_values, sources, feature_names),
        "unreviewed_labels": unreviewed_labels_check(setup),
        "label_sibling_exclusion": label_sibling_exclusion_check(setup),
    }


def raise_training_checks(
    checks_store: Any, run_id: str, report: dict[str, Any], step: str | None = "train_setup"
) -> list[str]:
    """Register every raised Check against the Training Run (deduped per run)."""
    raised: list[str] = []
    for kind in CHECK_KINDS:
        finding = report.get(kind) or {}
        if not finding.get("raised"):
            continue
        if checks_store.has_check(kind, "training_run", run_id):
            continue
        checks_store.register(
            kind=kind,
            severity=finding.get("severity") or "info",
            message=finding.get("message") or kind,
            subject_type="training_run",
            subject_id=run_id,
            details=finding.get("details") or {},
            step=step,
        )
        raised.append(kind)
    return raised
