"""On-demand evaluation and prediction for a fitted Model.

Nothing here runs automatically after a Training Run. A plot is computed when
the user asks for it, and a prediction when they ask for that — a Training Run
finishes in seconds and doing either eagerly would be work nobody asked for.

The one thing that must not drift is the **preprocessing**. A prediction applies
the exact `Preprocessor` the Model was trained with, stored on the Training Run,
via :func:`~dat_distiller.training.preprocess.transform_with_spec`. Re-deriving
the pipeline from the request would let a column set drift and quietly produce
predictions from a differently-transformed matrix.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np
import pandas as pd

from .checks import value_key
from .preprocess import Preprocessor, transform_with_spec
from .trainers import (
    FittedModel,
    ModelSpec,
    build_estimator,
    encode_classes,
    get_spec,
)

#: Columns added to a prediction download, after the input columns.
PREDICTION_COLUMN = "prediction"
PROBABILITY_PREFIX = "probability_"


class EvaluationError(ValueError):
    """A request that cannot be served — reported to the user, never a 500."""


class NotOnRunError(EvaluationError):
    """The Model is not on this Training Run (or did not fit).

    Split out from `EvaluationError` because it is a mistake in the *request* and
    deserves a 422, whereas a Model that exists but cannot produce a given plot
    deserves a 200 with a reason.
    """


# -- loading a fitted model ---------------------------------------------------


def _require_spec(name: str) -> ModelSpec:
    try:
        return get_spec(name)
    except KeyError as exc:
        raise NotOnRunError(
            f"unknown Model {name!r}; see GET /api/train/models for the list"
        ) from exc


def load_model(run: Mapping[str, Any], model_name: str) -> tuple[ModelSpec, FittedModel]:
    """Rebuild a fitted Model from a stored Training Run.

    A Training Run records *what it fitted*, not the estimator, so the estimator
    is rebuilt deterministically from the recorded hyperparameters and the
    recorded seed, then refitted on the recorded training split. Same recipe,
    same rows, same result — which is why the recorded `hyperparameters` and
    `seeds` have to travel with the run.
    """
    entries = {entry["model"]: entry for entry in run.get("leaderboard", [])}
    entry = entries.get(model_name)
    if entry is None:
        fitted = [name for name, e in entries.items() if e.get("status") == "ok"]
        raise NotOnRunError(
            f"Model {model_name!r} is not on this Training Run"
            + (f" (it fitted: {', '.join(sorted(fitted))})" if fitted else "")
        )
    if entry.get("status") != "ok":
        raise NotOnRunError(
            f"Model {model_name!r} did not fit on this Training Run: "
            f"{entry.get('error') or 'unknown error'}"
        )

    spec = _require_spec(model_name)
    task_type = str(entry.get("task_type") or run.get("task_type"))
    classes = entry.get("classes")
    estimator = build_estimator(
        spec, task_type, hyperparameters=entry.get("hyperparameters") or {}
    )
    return spec, FittedModel(
        spec=spec,
        task_type=task_type,
        estimator=estimator,
        classes=list(classes) if classes else [],
        hyperparameters=entry.get("hyperparameters") or {},
        fit_seconds=float(entry.get("fit_seconds") or 0.0),
    )


def _pipeline(run: Mapping[str, Any]) -> dict[str, Any]:
    spec = run.get("preprocessing")
    if not isinstance(spec, Mapping) or not spec:
        raise EvaluationError("this Training Run has no stored preprocessing to reuse")
    return dict(spec)


def _train_matrix(run: Mapping[str, Any], frame: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """The exact matrix this Model was trained on, rebuilt from the split."""
    rows = (run.get("training_split") or {}).get("rows")
    if rows is None:
        raise EvaluationError("this Training Run does not record its training split")
    indices = [int(i) for i in rows]
    if not indices:
        raise EvaluationError("this Training Run records an empty training split")
    matrix = transform_with_spec(frame, _pipeline(run))
    target = str(run.get("target"))
    if target not in frame.columns:
        raise EvaluationError(f"Target {target!r} is not in the data")
    if max(indices) >= len(frame):
        raise EvaluationError("the stored training split does not fit the data")
    values = frame[target].iloc[indices]
    if str(run.get("task_type")) == "classification":
        y, _ = encode_classes(values)
    else:
        y = np.asarray(pd.to_numeric(values, errors="coerce").fillna(0.0), dtype=float)
    return matrix[indices], y


def refit(run: Mapping[str, Any], model_name: str, frame: pd.DataFrame) -> FittedModel:
    """A stored Model, refitted on its recorded training split. See :func:`load_model`.

    Fitted on the *encoded* Target exactly as `trainers.fit_model` does during a
    Training Run, so `predict` returns class codes and `FittedModel.proba` can
    align its columns against the recorded `classes`.
    """
    _, fitted = load_model(run, model_name)
    X, y = _train_matrix(run, frame)
    fitted.estimator.fit(X, y)
    return fitted


# -- evaluation ---------------------------------------------------------------


def confusion_matrix_plot(
    y_true: np.ndarray, y_pred: np.ndarray, classes: list[str]
) -> dict[str, Any]:
    """Counts, plus a normalised view. Both use the SAME class order.

    An off-by-one in the axis order is the classic confusion-matrix bug, so the
    order is returned with the numbers and `labels` is what a chart must use.
    """
    labels = classes or sorted({str(v) for v in y_true} | {str(v) for v in y_pred})
    index = {label: i for i, label in enumerate(labels)}
    size = len(labels)
    matrix = np.zeros((size, size), dtype=int)
    for actual, predicted in zip(y_true, y_pred, strict=True):
        matrix[index[str(actual)], index[str(predicted)]] += 1
    totals = matrix.sum(axis=1, keepdims=True)
    normalised = np.divide(
        matrix, totals, out=np.zeros_like(matrix, dtype=float), where=totals > 0
    )
    return {
        "labels": labels,
        "matrix": matrix.tolist(),
        "row_normalised": [[round(float(v), 6) for v in row] for row in normalised],
        "support": matrix.sum(axis=1).tolist(),
        "n": int(matrix.sum()),
    }


def roc_curve_plot(
    y_true: np.ndarray, proba: np.ndarray, classes: list[str]
) -> dict[str, Any]:
    """ROC points for the positive class, plus the trapezoidal AUC.

    Binary uses the second column of the probability matrix; multiclass uses the
    one-vs-rest macro, and says which it did.
    """
    if proba is None:
        raise EvaluationError("this Model does not output class probabilities")
    if proba.ndim != 2 or proba.shape[1] != len(classes):
        raise EvaluationError(
            f"expected {len(classes)} probability column(s), got "
            f"{0 if proba.ndim != 2 else proba.shape[1]}"
        )
    if len(classes) < 2:
        # one class is not a ranking problem; say so rather than emit a
        # diagonal "curve" that looks meaningful
        raise EvaluationError(
            "a ROC curve needs at least two classes; this Target has one"
        )
    binary = len(classes) == 2
    if binary:
        scores = np.asarray(proba[:, 1], dtype=float)
        positive = classes[1]
        truth = np.asarray([1 if str(v) == positive else 0 for v in y_true], dtype=int)
        points, auc = _roc_binary(truth, scores)
        return {
            "kind": "binary",
            "positive_class": positive,
            "fpr": [round(p[0], 6) for p in points],
            "tpr": [round(p[1], 6) for p in points],
            "thresholds": [round(float(t), 6) for t in _thresholds(truth, scores)],
            "auc": round(float(auc), 6),
        }
    curves = {}
    aucs = []
    for i, label in enumerate(classes):
        truth = np.asarray([1 if str(v) == label else 0 for v in y_true], dtype=int)
        points, auc = _roc_binary(truth, np.asarray(proba[:, i], dtype=float))
        curves[label] = {
            "fpr": [round(p[0], 6) for p in points],
            "tpr": [round(p[1], 6) for p in points],
        }
        aucs.append(auc)
    return {
        "kind": "multiclass_ovr",
        "classes": list(classes),
        "curves": curves,
        "auc_macro": round(float(np.mean(aucs)), 6),
    }


def _thresholds(truth: np.ndarray, scores: np.ndarray) -> np.ndarray:
    return np.r_[np.inf, np.sort(np.unique(scores))[::-1]]


def _roc_binary(truth: np.ndarray, scores: np.ndarray) -> tuple[list[tuple[float, float]], float]:
    """ROC by sorting on score. Written out rather than imported so the curve is
    reproducible and does not depend on an sklearn version's internals."""
    order = np.argsort(-scores, kind="mergesort")
    t = truth[order]
    s = scores[order]
    positives = int(t.sum())
    negatives = int(t.size - positives)
    if positives == 0 or negatives == 0:
        # A single-class fold has no ROC. An empty curve beats a fabricated one.
        return [(0.0, 0.0), (1.0, 1.0)], float("nan")
    tps = np.cumsum(t)
    fps = np.cumsum(1 - t)
    # keep only the last index of each run of equal scores
    keep = np.r_[np.diff(s) != 0, True]
    tps, fps = tps[keep], fps[keep]
    points = [(0.0, 0.0)]
    points += [(float(f) / negatives, float(p) / positives) for p, f in zip(tps, fps, strict=True)]
    points.append((1.0, 1.0))
    auc = float(
        sum(
            (points[i + 1][0] - points[i][0]) * (points[i + 1][1] + points[i][1]) / 2
            for i in range(len(points) - 1)
        )
    )
    return points, auc


def precision_recall_plot(
    y_true: np.ndarray, proba: np.ndarray, classes: list[str]
) -> dict[str, Any]:
    if proba is None:
        raise EvaluationError("this Model does not output class probabilities")
    if len(classes) < 2:
        raise EvaluationError(
            "a precision-recall curve needs at least two classes; this Target has one"
        )
    binary = len(classes) == 2
    if binary:
        positive = classes[1]
        truth = np.asarray([1 if str(v) == positive else 0 for v in y_true], dtype=int)
        precision, recall, avg = _pr_binary(truth, np.asarray(proba[:, 1], dtype=float))
        return {
            "kind": "binary",
            "positive_class": positive,
            "precision": [round(p, 6) for p in precision],
            "recall": [round(r, 6) for r in recall],
            "average_precision": round(float(avg), 6),
        }
    curves = {}
    averages = []
    for i, label in enumerate(classes):
        truth = np.asarray([1 if str(v) == label else 0 for v in y_true], dtype=int)
        precision, recall, avg = _pr_binary(truth, np.asarray(proba[:, i], dtype=float))
        curves[label] = {
            "precision": [round(p, 6) for p in precision],
            "recall": [round(r, 6) for r in recall],
        }
        averages.append(avg)
    return {
        "kind": "multiclass_ovr",
        "classes": list(classes),
        "curves": curves,
        "average_precision_macro": round(float(np.mean(averages)), 6),
    }


def _pr_binary(
    truth: np.ndarray, scores: np.ndarray
) -> tuple[np.ndarray, np.ndarray, float]:
    order = np.argsort(-scores, kind="mergesort")
    t = truth[order]
    positives = int(t.sum())
    if positives == 0:
        return np.array([1.0, 0.0]), np.array([0.0, 0.0]), float("nan")
    tps = np.cumsum(t)
    fps = np.cumsum(1 - t)
    keep = np.r_[np.diff(scores[order]) != 0, True]
    tps, fps = tps[keep], fps[keep]
    precision = tps / np.maximum(tps + fps, 1)
    recall = tps / positives
    order_r = np.argsort(recall, kind="mergesort")
    p, r = precision[order_r], recall[order_r]
    # average precision == area under the PR curve, by the step rule
    ap = float(np.sum(np.diff(np.r_[0.0, r]) * p))
    return np.r_[p, 0.0], np.r_[r, 1.0], ap


def residual_plots(
    y_true: np.ndarray, y_pred: np.ndarray
) -> dict[str, Any]:
    residuals = np.asarray(y_true, dtype=float) - np.asarray(y_pred, dtype=float)
    return {
        "residuals": [round(float(v), 6) for v in residuals],
        "predicted": [round(float(v), 6) for v in y_pred],
        "actual": [round(float(v), 6) for v in y_true],
        "mae": round(float(np.abs(residuals).mean()), 6),
        "mean_residual": round(float(residuals.mean()), 6),
        "n": int(residuals.size),
    }


def permutation_importance(
    model: FittedModel,
    X: np.ndarray,
    y: np.ndarray,
    feature_names: list[str],
    *,
    repeats: int = 5,
    seed: int = 0,
) -> dict[str, Any]:
    """Permutation importance: how much a column's accuracy drops when shuffled.

    Works for every estimator — including ones with no native importance — which
    is why it is the fallback. Uses the Model's own scoring rule so the number
    means what the leaderboard means.
    """
    if not feature_names:
        raise EvaluationError("no feature names to attribute importance to")
    if X.shape[0] < 2:
        raise EvaluationError("permutation importance needs at least two rows")
    rng = np.random.default_rng(seed)
    baseline = _score(model, X, y)
    scores: dict[str, float] = {}
    for index, name in enumerate(feature_names):
        drops = []
        for _ in range(max(1, repeats)):
            shuffled = X.copy()
            shuffled[:, index] = rng.permutation(shuffled[:, index])
            drops.append(baseline - _score(model, shuffled, y))
        scores[name] = round(float(np.mean(drops)), 6)
    ordered = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)
    return {
        "method": "permutation",
        "baseline": round(float(baseline), 6),
        "repeats": repeats,
        "importance": [{"feature": name, "drop": value} for name, value in ordered],
    }


def _score(model: FittedModel, X: np.ndarray, y: np.ndarray) -> float:
    """Accuracy for classification, R² for regression — the Model's own metrics.

    `evaluate_classification` wants class *codes*, which is exactly what the
    estimator was fitted on and what `predict` returns, so `y` is passed through
    untouched. Mapping codes back to labels first is a bug: the metric function
    casts them to int again.
    """
    from .trainers import evaluate_classification, evaluate_regression

    predicted = model.predict(X)
    if model.task_type == "classification":
        return float(
            evaluate_classification(np.asarray(y, dtype="int64"), np.asarray(predicted, dtype="int64"), classes=model.classes)["accuracy"]["value"]
            or 0.0
        )
    return float(evaluate_regression(y, predicted)["r2"]["value"] or 0.0)


# -- prediction ---------------------------------------------------------------


def _readable_class(key: str) -> str:
    """A class name as a person reads it.

    Classes are stored as `value_key` strings (`bool:True`, `int:3`) so a
    nullable boolean Target round-trips through JSON. That is the right thing to
    store and the wrong thing to show a user reading a prediction column, so the
    prefix is stripped on the way out.
    """
    prefix, _, rest = key.partition(":")
    if prefix in {"bool", "int", "num", "str"} and rest != "":
        return rest
    return key


def predict_frame(
    run: Mapping[str, Any],
    model_name: str,
    frame: pd.DataFrame,
    model: FittedModel,
) -> pd.DataFrame:
    """The input frame plus a prediction column (and probabilities).

    Missing required columns are refused with a message naming *which* ones, not
    "invalid input" — a user who left out a column needs to know which.
    """
    pipeline = Preprocessor.from_dict(_pipeline(run))
    missing = [column for column in pipeline.spec.features if column not in frame.columns]
    if missing:
        raise EvaluationError(
            f"this data is missing column(s) the Model needs: {', '.join(missing)}. "
            f"It was trained on {len(pipeline.spec.features)} column(s)."
        )
    if len(frame) == 0:
        raise EvaluationError("there are no rows to predict")

    X = transform_with_spec(frame, pipeline.to_dict())
    out = frame.copy()
    predicted = model.predict(X)
    if model.task_type == "classification":
        if model.classes:
            out[PREDICTION_COLUMN] = [_readable_class(model.classes[int(i)]) for i in predicted]
        else:
            out[PREDICTION_COLUMN] = [str(v) for v in predicted]
        proba = model.proba(X)
        if proba is not None:
            for index, label in enumerate(model.classes or []):
                out[f"{PROBABILITY_PREFIX}{_readable_class(label)}"] = [
                    round(float(v), 6) for v in np.asarray(proba)[:, index]
                ]
    else:
        out[PREDICTION_COLUMN] = [round(float(v), 6) for v in np.asarray(predicted, dtype=float)]
    return out


def evaluate_run(
    run: Mapping[str, Any],
    model_name: str,
    frame: pd.DataFrame,
    *,
    plot: str,
    repeats: int = 5,
) -> dict[str, Any]:
    """One on-demand plot for one Model. Never runs unless this is called."""
    if plot not in PLOTS:
        raise EvaluationError(f"plot must be one of {PLOTS}, got {plot!r}")
    if plot == "feature_importance":
        # permutation importance needs a fitted estimator, so refit like every
        # other plot rather than scoring an unfitted one
        model = refit(run, model_name, frame)
        names = list(run.get("feature_names") or [])
        return {
            "plot": plot,
            "model": model_name,
            **permutation_importance(
                model,
                *_train_matrix(run, frame),
                names,
                repeats=repeats,
                seed=int((run.get("seeds") or {}).get("models") or run.get("seed") or 0),
            ),
        }

    model = refit(run, model_name, frame)
    test_rows = [int(i) for i in (run.get("test_split") or {}).get("indices") or []]
    if not test_rows:
        raise EvaluationError("this Training Run records no held-out test split")
    if max(test_rows) >= len(frame):
        raise EvaluationError("the stored test split does not fit the data")
    X = transform_with_spec(frame, _pipeline(run))[test_rows]
    target = str(run.get("target"))
    raw = frame[target].iloc[test_rows]
    if str(run.get("task_type")) == "classification":
        # Both sides of a classification plot must speak the same label language.
        # `model.classes` are `value_key` strings and `predict` returns codes, so
        # the truth is keyed and the prediction mapped back before plotting —
        # otherwise a nullable-boolean Target indexes a label list it never had.
        truth = [value_key(v) for v in raw]
        predicted_keys = [
            model.classes[int(i)] if 0 <= int(i) < len(model.classes) else f"unknown:{i}"
            for i in model.predict(X)
        ]
        if plot == "confusion_matrix":
            return {
                "plot": plot,
                "model": model_name,
                **confusion_matrix_plot(truth, predicted_keys, model.classes),
            }
        if plot == "roc":
            return {
                "plot": plot,
                "model": model_name,
                **roc_curve_plot(truth, model.proba(X), model.classes),
            }
        return {
            "plot": plot,
            "model": model_name,
            **precision_recall_plot(truth, model.proba(X), model.classes),
        }
    y = np.asarray(pd.to_numeric(raw, errors="coerce").fillna(0.0), dtype=float)
    return {"plot": plot, "model": model_name, **residual_plots(y, model.predict(X))}


#: The on-demand plots, and the Task Type each one belongs to.
PLOTS = ("confusion_matrix", "roc", "precision_recall", "residuals", "feature_importance")
PLOT_TASK_TYPES = {
    "confusion_matrix": "classification",
    "roc": "classification",
    "precision_recall": "classification",
    "residuals": "regression",
    "feature_importance": "classification+regression",
}


__all__ = [
    "PLOTS",
    "PREDICTION_COLUMN",
    "PROBABILITY_PREFIX",
    "EvaluationError",
    "NotOnRunError",
    "confusion_matrix_plot",
    "evaluate_run",
    "load_model",
    "permutation_importance",
    "precision_recall_plot",
    "predict_frame",
    "refit",
    "residual_plots",
    "roc_curve_plot",
]
