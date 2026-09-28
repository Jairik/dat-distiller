"""The Model registry, the fitting, and the metrics behind the leaderboard.

One place that knows what a **Model** is, which Task Types it supports, how to
build it with its defaults, how to tune it, and how to score it on the held-out
test split.

Three pieces, deliberately separate:

- **The registry** (:data:`MODEL_SPECS`) — each :class:`ModelSpec` declares the
  Task Types it supports, the library and optional extra it comes from, its
  default hyperparameters, and a small search space. A Model that does not
  support the Target's Task Type (``naive_bayes`` and regression) is refused
  with a readable error rather than silently fitted the wrong way.
- **The matrices** (:class:`TrainingMatrix`) — tuning can only ever be handed a
  ``TrainingMatrix``, and the only way to build one is
  :meth:`TrainingMatrix.from_split`, which reads the *training* split and
  nothing else. The test split is structurally out of reach during tuning: it is
  not a type the search accepts, and the only constructor ignores the held-out
  indices entirely.
- **The metrics** (:data:`METRICS`) — accuracy / F1 / ROC-AUC and friends for
  classification, MAE / RMSE / R² and friends for regression. A metric that
  cannot be computed honestly (no class probabilities, a single-class test
  split) is reported as ``null`` **with a reason**; a number is never invented.

scikit-learn and LightGBM are optional extras: they are imported lazily, and
:func:`available_models` / :func:`library_versions` say plainly when the
install cannot run them.
"""

from __future__ import annotations

import sys
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from importlib import import_module
from importlib.metadata import PackageNotFoundError
from importlib.metadata import version as dist_version
from typing import Any

import numpy as np

from ..extras import is_installed
from .checks import value_key

CLASSIFICATION = "classification"
REGRESSION = "regression"
TASK_TYPES = (CLASSIFICATION, REGRESSION)

#: Library name -> distribution name, for the versions a Training Run records.
LIBRARY_DISTRIBUTIONS = {
    "scikit-learn": "scikit-learn",
    "lightgbm": "lightgbm",
}

#: What every :class:`TrainingMatrix` must be built with. Without it the
#: constructor refuses, so the only path to a fitted search is
#: :meth:`TrainingMatrix.from_split` — which only ever reads the training split.
_MATRIX_KEY = "dat_distiller.training.trainers.TrainingMatrix"


class ModelError(ValueError):
    """A Model request the Training Run cannot honour (bad name or Task Type)."""


class UnsupportedModelError(ModelError):
    """The Model does not support the Target's Task Type."""


class ModelUnavailableError(ModelError):
    """The Model's library is an optional extra this install does not have."""


# -- the Target's values -----------------------------------------------------


def encode_classes(values: Sequence[Any]) -> tuple[np.ndarray, list[str]]:
    """Class-encode the Target as ``0..k-1`` and name the classes with
    :func:`value_key`, so booleans, nullable booleans, strings and numbers all
    behave the same and a persisted leaderboard round-trips as JSON."""
    keys = [value_key(value) for value in values]
    classes = sorted(set(keys))
    lookup = {key: position for position, key in enumerate(classes)}
    return np.array([lookup[key] for key in keys], dtype="int64"), classes


def encode_regression(values: Sequence[Any]) -> np.ndarray:
    """The Target as floats; anything not a number is a readable refusal."""
    out: list[float] = []
    for index, value in enumerate(values):
        try:
            out.append(float(value))
        except (TypeError, ValueError) as exc:
            raise ModelError(
                f"Target value at row {index} ({value!r}) is not a number, so this "
                "Target cannot be a regression Target; set the Task Type to "
                "classification instead"
            ) from exc
    return np.asarray(out, dtype="float64")


# -- the registry ------------------------------------------------------------


@dataclass(frozen=True)
class ModelSpec:
    """One Model: what it is, what it can do, and how it is built."""

    name: str
    label: str
    library: str
    extra: str
    task_types: tuple[str, ...]
    #: ``{"classification": "sklearn.svm:SVC", "regression": "sklearn.svm:SVR"}``
    estimators: dict[str, str]
    #: Whether the Model's *default* configuration outputs class probabilities.
    #: A user override can still switch it on; :meth:`FittedModel.proba` asks
    #: the fitted estimator, and only :func:`tune_model` consults this flag.
    supports_proba: bool
    #: Defaults per Task Type; the ``"*"`` key applies to all of them. They are
    #: per Task Type because one Model name is often two estimator classes with
    #: different parameters (SVC and SVR, say).
    default_hyperparameters: dict[str, dict[str, Any]] = field(default_factory=dict)
    #: Hyperparameters that receive the Training Run's seed, per Task Type for
    #: the same reason: ``SVC`` takes ``random_state``, ``SVR`` does not.
    seeded_params: dict[str, tuple[str, ...]] = field(default_factory=dict)
    #: A small grid for optional tuning. Deliberately tiny: the leaderboard
    #: must stay quick enough to re-run from the UI.
    search_space: dict[str, list[Any]] = field(default_factory=dict)
    notes: str = ""

    def supports(self, task_type: str) -> bool:
        return task_type in self.task_types

    def path(self, task_type: str) -> str:
        try:
            return self.estimators[task_type]
        except KeyError as exc:  # pragma: no cover - guarded by supports()
            raise UnsupportedModelError(
                f"Model {self.name!r} does not support a {task_type} Target"
            ) from exc

    def defaults_for(self, task_type: str) -> dict[str, Any]:
        """The default hyperparameters for one Task Type."""
        return {
            **self.default_hyperparameters.get("*", {}),
            **self.default_hyperparameters.get(task_type, {}),
        }

    def seeds_for(self, task_type: str) -> tuple[str, ...]:
        """Which hyperparameters receive the Training Run's seed for a Task Type."""
        if task_type in self.seeded_params:
            return tuple(self.seeded_params[task_type])
        return tuple(self.seeded_params.get("*", ()))

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "label": self.label,
            "library": self.library,
            "extra": self.extra,
            "task_types": list(self.task_types),
            "supports_proba": bool(self.supports_proba),
            "default_hyperparameters": {
                task_type: self.defaults_for(task_type) for task_type in self.task_types
            },
            "seeded_params": {
                task_type: list(self.seeds_for(task_type)) for task_type in self.task_types
            },
            "search_space": {k: list(v) for k, v in self.search_space.items()},
            "notes": self.notes,
        }


MODEL_SPECS: dict[str, ModelSpec] = {
    spec.name: spec
    for spec in (
        ModelSpec(
            name="logistic_regression",
            label="Logistic regression",
            library="scikit-learn",
            extra="sklearn",
            task_types=(CLASSIFICATION,),
            estimators={"classification": "sklearn.linear_model:LogisticRegression"},
            supports_proba=True,
            default_hyperparameters={"*": {"max_iter": 1000}},
            seeded_params={"*": ("random_state",)},
            search_space={"C": [0.01, 0.1, 1.0, 10.0]},
        ),
        ModelSpec(
            name="linear_regression",
            label="Linear regression",
            library="scikit-learn",
            extra="sklearn",
            task_types=(REGRESSION,),
            estimators={"regression": "sklearn.linear_model:LinearRegression"},
            supports_proba=False,
            search_space={"fit_intercept": [True, False]},
            notes="least squares; deterministic, so it takes no seed",
        ),
        ModelSpec(
            name="svm",
            label="SVM",
            library="scikit-learn",
            extra="sklearn",
            task_types=(CLASSIFICATION, REGRESSION),
            estimators={
                "classification": "sklearn.svm:SVC",
                "regression": "sklearn.svm:SVR",
            },
            supports_proba=False,
            default_hyperparameters={"*": {"C": 1.0, "kernel": "rbf"}},
            # SVR is deterministic and has no random_state; SVC needs one to
            # calibrate probabilities, so only the classifier is seeded.
            seeded_params={"classification": ("random_state",)},
            search_space={"C": [0.1, 1.0, 10.0], "gamma": ["scale", 0.01, 0.1]},
            notes=(
                "no class probabilities by default, so ROC-AUC is null; override "
                'with {"probability": true} to calibrate them'
            ),
        ),
        ModelSpec(
            name="random_forest",
            label="Random Forest",
            library="scikit-learn",
            extra="sklearn",
            task_types=(CLASSIFICATION, REGRESSION),
            estimators={
                "classification": "sklearn.ensemble:RandomForestClassifier",
                "regression": "sklearn.ensemble:RandomForestRegressor",
            },
            supports_proba=True,
            default_hyperparameters={"*": {"n_estimators": 100}},
            seeded_params={"*": ("random_state",)},
            search_space={"n_estimators": [50, 100], "max_depth": [None, 5, 10]},
        ),
        ModelSpec(
            name="gradient_boosting",
            label="Gradient boosting",
            library="scikit-learn",
            extra="sklearn",
            task_types=(CLASSIFICATION, REGRESSION),
            estimators={
                "classification": "sklearn.ensemble:GradientBoostingClassifier",
                "regression": "sklearn.ensemble:GradientBoostingRegressor",
            },
            supports_proba=True,
            default_hyperparameters={"*": {"n_estimators": 100}},
            seeded_params={"*": ("random_state",)},
            search_space={
                "n_estimators": [50, 100],
                "learning_rate": [0.05, 0.1, 0.3],
            },
        ),
        ModelSpec(
            name="knn",
            label="k-NN",
            library="scikit-learn",
            extra="sklearn",
            task_types=(CLASSIFICATION, REGRESSION),
            estimators={
                "classification": "sklearn.neighbors:KNeighborsClassifier",
                "regression": "sklearn.neighbors:KNeighborsRegressor",
            },
            supports_proba=True,
            default_hyperparameters={"*": {"n_neighbors": 5}},
            search_space={"n_neighbors": [1, 3, 5, 7]},
            notes="deterministic, so it takes no seed",
        ),
        ModelSpec(
            name="naive_bayes",
            label="Naive Bayes",
            library="scikit-learn",
            extra="sklearn",
            task_types=(CLASSIFICATION,),
            estimators={"classification": "sklearn.naive_bayes:GaussianNB"},
            supports_proba=True,
            search_space={"var_smoothing": [1e-9, 1e-8, 1e-7]},
            notes="classification only",
        ),
        ModelSpec(
            name="lightgbm",
            label="LightGBM",
            library="lightgbm",
            extra="lightgbm",
            task_types=(CLASSIFICATION, REGRESSION),
            estimators={
                "classification": "lightgbm:LGBMClassifier",
                "regression": "lightgbm:LGBMRegressor",
            },
            supports_proba=True,
            default_hyperparameters={"*": {"n_estimators": 100, "verbosity": -1, "n_jobs": 1}},
            seeded_params={"*": ("random_state",)},
            search_space={"num_leaves": [7, 15, 31], "learning_rate": [0.05, 0.1]},
        ),
    )
}

#: Stable order for the leaderboard and the model picker.
MODEL_NAMES: tuple[str, ...] = tuple(MODEL_SPECS)


def get_spec(name: str) -> ModelSpec:
    try:
        return MODEL_SPECS[str(name)]
    except KeyError as exc:
        raise ModelError(
            f"unknown Model {name!r}; available Models: {', '.join(MODEL_NAMES)}"
        ) from exc


def is_available(spec: ModelSpec) -> bool:
    """Whether this install can actually build the Model (extra present)."""
    return is_installed(spec.extra)


def library_versions(names: Sequence[str] | None = None) -> dict[str, str | None]:
    """Versions of the libraries a Training Run records, ``None`` when absent."""
    out: dict[str, str | None] = {"python": ".".join(str(p) for p in sys.version_info[:3])}
    for name in names if names is not None else list(LIBRARY_DISTRIBUTIONS):
        try:
            out[name] = dist_version(LIBRARY_DISTRIBUTIONS[name])
        except PackageNotFoundError:
            out[name] = None
    return out


def available_models(task_type: str | None = None) -> list[str]:
    """Model names for a Task Type, or all of them, that this install can run."""
    return [
        name
        for name, spec in MODEL_SPECS.items()
        if (task_type is None or spec.supports(task_type)) and is_available(spec)
    ]


def resolve_models(names: Sequence[str] | None, task_type: str) -> list[ModelSpec]:
    """Turn the user's selection into ModelSpecs, refusing impossible pairings.

    ``None`` means every Model this install can run for ``task_type``. An
    explicit name must exist, must support the Task Type, and must be installed
    — each refusal names what to do about it.
    """
    if task_type not in TASK_TYPES:
        raise ModelError(f"task type must be one of {TASK_TYPES}")
    if names is None:
        chosen = [MODEL_SPECS[name] for name in available_models(task_type)]
        if not chosen:
            raise ModelUnavailableError(
                "no Model is installed for this Task Type; install the optional "
                "extras with `uv sync --extra sklearn --extra lightgbm`"
            )
        return chosen
    if not names:
        raise ModelError("select at least one Model to train")
    supported = [n for n, s in MODEL_SPECS.items() if s.supports(task_type)]
    resolved: list[ModelSpec] = []
    for name in names:
        spec = get_spec(name)
        if not spec.supports(task_type):
            raise UnsupportedModelError(
                f"Model {spec.name!r} ({spec.label}) supports only "
                f"{' and '.join(spec.task_types)}, but this Target's Task Type is "
                f"{task_type}. Models for a {task_type} Target: {', '.join(supported)}"
            )
        if not is_available(spec):
            raise ModelUnavailableError(
                f"Model {spec.name!r} needs the optional extra {spec.extra!r}, which "
                "is not installed; install it with "
                f"`uv sync --extra {spec.extra}` or drop the Model"
            )
        resolved.append(spec)
    return resolved


# -- building one Model ------------------------------------------------------


def _import_estimator(path: str) -> type:
    module_name, _, attribute = path.partition(":")
    try:
        module = import_module(module_name)
    except ImportError as exc:
        raise ModelUnavailableError(
            f"{module_name} is not installed; it is an optional extra of Dat Distiller"
        ) from exc
    return getattr(module, attribute)


def resolve_hyperparameters(
    spec: ModelSpec,
    task_type: str,
    *,
    seed: int,
    overrides: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Defaults, plus the seed where the Model takes one, plus the user's
    overrides — the exact mapping recorded on the Training Run."""
    resolved: dict[str, Any] = dict(spec.defaults_for(task_type))
    for name in spec.seeds_for(task_type):
        resolved[name] = int(seed)
    resolved.update(dict(overrides or {}))
    return resolved


def build_estimator(
    spec: ModelSpec, task_type: str, *, hyperparameters: Mapping[str, Any] | None = None
) -> Any:
    """Instantiate the Model's estimator for a Task Type."""
    if not is_available(spec):
        raise ModelUnavailableError(
            f"Model {spec.name!r} needs the optional extra {spec.extra!r}, which is not installed"
        )
    cls = _import_estimator(spec.path(task_type))
    return cls(**dict(hyperparameters or {}))


# -- the training matrix -----------------------------------------------------


class TrainingMatrix:
    """The training split's matrix and Target values — the only thing a search sees.

    Construct one with :meth:`from_split`; the constructor is closed so that a
    caller cannot hand a search a matrix built from the held-out test rows.
    """

    __slots__ = ("X", "classes", "feature_names", "rows", "task_type", "y")

    def __init__(
        self,
        X: np.ndarray,
        y: np.ndarray,
        *,
        feature_names: Sequence[str],
        task_type: str,
        rows: Sequence[int],
        classes: list[str],
        key: str | None = None,
    ) -> None:
        if key != _MATRIX_KEY:
            raise TypeError(
                "a TrainingMatrix can only be built with TrainingMatrix.from_split(...), "
                "which reads the training split of a held-out split"
            )
        self.X = np.ascontiguousarray(X, dtype="float64")
        self.y = np.ascontiguousarray(y)
        self.feature_names = [str(name) for name in feature_names]
        self.task_type = str(task_type)
        self.rows = [int(row) for row in rows]
        self.classes = list(classes)

    @classmethod
    def from_split(
        cls,
        *,
        frame: Any,
        spec: Mapping[str, Any] | Any,
        setup: Any,
    ) -> TrainingMatrix:
        """The matrix for ``setup.split.train`` and nothing else.

        ``spec`` is the fitted preprocessing pipeline from #31 (a
        :class:`~dat_distiller.training.preprocess.Preprocessor` or its persisted
        dict, applied with ``transform_with_spec``). The held-out indices are
        never read here, which is what makes tuning leak-free by construction.
        """
        from .preprocess import (
            transform_with_spec,  # local: keeps the import graph flat
        )

        train_frame = frame.iloc[list(setup.split.train)]
        values = train_frame[setup.target].to_numpy()
        matrix = transform_with_spec(train_frame, spec)
        if setup.task_type == CLASSIFICATION:
            y, classes = encode_classes(values)
        else:
            y, classes = encode_regression(values), []
        return cls(
            matrix,
            y,
            feature_names=_spec_feature_names(spec),
            task_type=setup.task_type,
            rows=list(setup.split.train),
            classes=classes,
            key=_MATRIX_KEY,
        )

    @property
    def n_samples(self) -> int:
        return int(self.X.shape[0])

    @property
    def n_features(self) -> int:
        return int(self.X.shape[1])

    def __len__(self) -> int:  # pragma: no cover - trivial
        return self.n_samples

    def to_dict(self) -> dict[str, Any]:
        return {
            "task_type": self.task_type,
            "rows": list(self.rows),
            "n_samples": self.n_samples,
            "n_features": self.n_features,
            "classes": list(self.classes),
        }


def _spec_feature_names(spec: Mapping[str, Any] | Any) -> list[str]:
    """The transformed column names, from a Preprocessor or its persisted dict."""
    names = getattr(spec, "feature_names", None)
    if names is None and isinstance(spec, Mapping):
        names = spec.get("output_names")
    return [str(name) for name in (names or [])]


def assert_held_out(rows: Sequence[int], held_out: Sequence[int]) -> None:
    """Refuse a Training Matrix that overlaps the held-out test rows."""
    overlap = sorted(set(int(r) for r in rows) & set(int(r) for r in held_out))
    if overlap:
        raise AssertionError(
            f"the tuning matrix must not contain held-out test rows; {len(overlap)} overlap "
            f"at positions {overlap[:5]}"
        )


# -- fitting and scoring -----------------------------------------------------


@dataclass
class FittedModel:
    """A fitted estimator plus everything needed to score and re-fit it."""

    spec: ModelSpec
    task_type: str
    estimator: Any
    classes: list[str]
    hyperparameters: dict[str, Any]
    fit_seconds: float = 0.0

    def predict(self, matrix: np.ndarray) -> np.ndarray:
        return np.asarray(self.estimator.predict(matrix))

    def proba(self, matrix: np.ndarray) -> np.ndarray | None:
        """Class probabilities aligned to :attr:`classes`, or ``None``.

        ``None`` is an honest answer for a Model with no ``predict_proba``; the
        caller reports a null metric with a reason rather than a fake number.
        The registry's ``supports_proba`` flag describes the *default*
        configuration, so what counts here is what the fitted estimator can do.
        """
        if not hasattr(self.estimator, "predict_proba"):
            return None
        raw = np.asarray(self.estimator.predict_proba(matrix), dtype="float64")
        if raw.ndim != 2 or raw.shape[1] != len(self.classes):
            return None
        # Reindex defensively: the columns follow the estimator's own classes_.
        own = list(getattr(self.estimator, "classes_", []))
        if own == list(range(len(self.classes))):
            return raw
        lookup = {int(value): position for position, value in enumerate(own)}
        out = np.zeros_like(raw)
        for position, key in enumerate(self.classes):
            source = lookup.get(position)
            if source is not None:
                out[:, position] = raw[:, source]
        return out

    def to_dict(self) -> dict[str, Any]:
        return {
            "model": self.spec.name,
            "label": self.spec.label,
            "library": self.spec.library,
            "task_type": self.task_type,
            "classes": list(self.classes),
            "hyperparameters": dict(self.hyperparameters),
            "fit_seconds": round(float(self.fit_seconds), 4),
        }


def fit_model(
    spec: ModelSpec,
    task_type: str,
    matrix: np.ndarray,
    values: np.ndarray,
    *,
    hyperparameters: Mapping[str, Any] | None = None,
    classes: Sequence[str] | None = None,
) -> FittedModel:
    """Fit one Model with its defaults (plus any overrides) and time it."""
    estimator = build_estimator(spec, task_type, hyperparameters=hyperparameters)
    started = time.perf_counter()
    estimator.fit(matrix, values)
    elapsed = time.perf_counter() - started
    return FittedModel(
        spec=spec,
        task_type=task_type,
        estimator=estimator,
        classes=[str(c) for c in (classes or [])],
        hyperparameters=dict(hyperparameters or {}),
        fit_seconds=elapsed,
    )


# -- metrics -----------------------------------------------------------------


@dataclass(frozen=True)
class MetricSpec:
    """One leaderboard metric."""

    name: str
    label: str
    task_type: str
    higher_is_better: bool = True
    #: The metric is only defined for Models that output class probabilities.
    needs_proba: bool = False
    #: The default primary metric for this Task Type.
    is_default: bool = False
    #: scikit-learn scorer name used when the metric drives a search. ``None``
    #: means this module supplies the scorer itself — ROC-AUC has to know whether
    #: a fold is binary, and scikit-learn's own does not.
    scorer: str | None = None
    description: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "label": self.label,
            "task_type": self.task_type,
            "higher_is_better": bool(self.higher_is_better),
            "needs_proba": bool(self.needs_proba),
            "default": bool(self.is_default),
            "scorer": self.scorer,
            "description": self.description,
        }


METRICS: dict[str, MetricSpec] = {
    spec.name: spec
    for spec in (
        MetricSpec(
            "accuracy",
            "Accuracy",
            CLASSIFICATION,
            scorer="accuracy",
            description="share of held-out rows classified correctly",
        ),
        MetricSpec(
            "balanced_accuracy",
            "Balanced accuracy",
            CLASSIFICATION,
            scorer="balanced_accuracy",
            description="mean of the per-class recalls",
        ),
        MetricSpec(
            "f1_macro",
            "F1 (macro)",
            CLASSIFICATION,
            is_default=True,
            scorer="f1_macro",
            description="unweighted mean of the per-class F1 scores",
        ),
        MetricSpec(
            "f1_weighted",
            "F1 (weighted)",
            CLASSIFICATION,
            scorer="f1_weighted",
            description="per-class F1 weighted by class support",
        ),
        MetricSpec(
            "f1_minority",
            "F1 (smallest class)",
            CLASSIFICATION,
            description="F1 of the least frequent class in the test split",  # scorer below
        ),
        MetricSpec(
            "precision_macro",
            "Precision (macro)",
            CLASSIFICATION,
            scorer="precision_macro",
            description="unweighted mean of the per-class precisions",
        ),
        MetricSpec(
            "recall_macro",
            "Recall (macro)",
            CLASSIFICATION,
            scorer="recall_macro",
            description="unweighted mean of the per-class recalls",
        ),
        MetricSpec(
            "roc_auc",
            "ROC-AUC",
            CLASSIFICATION,
            needs_proba=True,
            description="one-vs-rest, macro-averaged when the Target has more than two classes",
        ),
        MetricSpec(
            "mae",
            "MAE",
            REGRESSION,
            higher_is_better=False,
            scorer="neg_mean_absolute_error",
            description="mean absolute error in the Target's own units",
        ),
        MetricSpec(
            "rmse",
            "RMSE",
            REGRESSION,
            higher_is_better=False,
            scorer="neg_root_mean_squared_error",
            description="root mean squared error; punishes large misses",
        ),
        MetricSpec(
            "r2",
            "R²",
            REGRESSION,
            is_default=True,
            scorer="r2",
            description="share of the held-out variance explained; 1.0 is perfect",
        ),
        MetricSpec(
            "median_absolute_error",
            "Median absolute error",
            REGRESSION,
            higher_is_better=False,
            scorer="neg_median_absolute_error",
            description="robust to a few very large misses",
        ),
        MetricSpec(
            "explained_variance",
            "Explained variance",
            REGRESSION,
            scorer="explained_variance",
            description="share of the held-out variance accounted for",
        ),
        MetricSpec(
            "max_error",
            "Max error",
            REGRESSION,
            higher_is_better=False,
            scorer="neg_max_error",
            description="the single worst prediction on the test split",
        ),
    )
}


def metrics_for(task_type: str) -> list[str]:
    """The metric names a Task Type can produce, default primary first."""
    names = [name for name, spec in METRICS.items() if spec.task_type == task_type]
    names.sort(key=lambda name: (not METRICS[name].is_default, name))
    return names


def default_primary_metric(task_type: str) -> str:
    """The metric a leaderboard ranks by unless the user picks another."""
    if task_type not in TASK_TYPES:
        raise ModelError(f"task type must be one of {TASK_TYPES}")
    for name, spec in METRICS.items():
        if spec.task_type == task_type and spec.is_default:
            return name
    raise ModelError(f"no default metric for a {task_type} Task Type")


def resolve_primary_metric(metric: str | None, task_type: str) -> str:
    """The user's primary metric, validated against what the Task Type produces."""
    if task_type not in TASK_TYPES:
        raise ModelError(f"task type must be one of {TASK_TYPES}")
    available = metrics_for(task_type)
    if metric is None:
        return default_primary_metric(task_type)
    if metric not in METRICS:
        raise ModelError(
            f"unknown metric {metric!r}; a {task_type} Target can be scored with: "
            f"{', '.join(available)}"
        )
    if METRICS[metric].task_type != task_type:
        raise ModelError(
            f"metric {metric!r} is a {METRICS[metric].task_type} metric, but this "
            f"Target's Task Type is {task_type}; a {task_type} Target can be scored "
            f"with: {', '.join(available)}"
        )
    return metric


def metric_value(metrics: Mapping[str, Any], name: str) -> float | None:
    """The number behind a metric entry, or ``None`` when it was not computable."""
    entry = metrics.get(name) or {}
    value = entry.get("value")
    return None if value is None else float(value)


def _measured(value: Any) -> dict[str, Any]:
    number = float(value)
    if not np.isfinite(number):
        return {"value": None, "reason": "the computation was not finite"}
    return {"value": round(number, 6), "reason": None}


def _missing(reason: str) -> dict[str, Any]:
    return {"value": None, "reason": reason}


def evaluate_classification(
    y_true: Sequence[Any],
    y_pred: Sequence[Any],
    *,
    proba: np.ndarray | None = None,
    classes: Sequence[str] | None = None,
) -> dict[str, Any]:
    """Classification metrics on the held-out test split.

    ``y_true`` and ``y_pred`` are class *codes* (``0..k-1``). A metric that
    cannot be computed on this test split — a single class, no probabilities —
    comes back ``null`` with the reason spelled out.
    """
    from sklearn import metrics as skm  # optional extra, imported lazily

    truth = np.asarray(y_true, dtype="int64")
    predicted = np.asarray(y_pred, dtype="int64")
    labels = list(range(len(classes))) if classes else sorted(
        set(truth.tolist()) | set(predicted.tolist())
    )
    present = sorted(set(truth.tolist()))
    out: dict[str, Any] = {}
    one_class = len(present) < 2

    out["accuracy"] = _measured(skm.accuracy_score(truth, predicted))
    if one_class:
        why = (
            f"the test split holds a single class ({classes[present[0]] if classes else present[0]}), "
            "so there is nothing to distinguish"
        )
    else:
        why = "the test split holds a single class, so there is nothing to distinguish"
    out["balanced_accuracy"] = (
        _measured(skm.balanced_accuracy_score(truth, predicted)) if not one_class else _missing(why)
    )
    out["f1_macro"] = (
        _measured(skm.f1_score(truth, predicted, average="macro", zero_division=0))
        if not one_class
        else _missing(why)
    )
    out["f1_weighted"] = (
        _measured(skm.f1_score(truth, predicted, average="weighted", zero_division=0))
        if not one_class
        else _missing(why)
    )
    out["precision_macro"] = (
        _measured(skm.precision_score(truth, predicted, average="macro", zero_division=0))
        if not one_class
        else _missing(why)
    )
    out["recall_macro"] = (
        _measured(skm.recall_score(truth, predicted, average="macro", zero_division=0))
        if not one_class
        else _missing(why)
    )
    if one_class:
        out["f1_minority"] = _missing(why)
    else:
        smallest = min(present, key=lambda code: int((truth == code).sum()))
        out["f1_minority"] = _measured(
            skm.f1_score(truth, predicted, labels=[smallest], average="macro", zero_division=0)
        )
    out["roc_auc"] = _roc_auc(skm, truth, proba, present, one_class, classes)
    out["confusion_matrix"] = skm.confusion_matrix(truth, predicted, labels=labels).tolist()
    out["n_test"] = len(truth)
    out["classes"] = [classes[code] if classes and code < len(classes) else str(code) for code in labels]
    out["support"] = {str(classes[c] if classes and c < len(classes) else c): int((truth == c).sum()) for c in present}
    return out


def _roc_auc(skm: Any, truth: np.ndarray, proba: np.ndarray | None, present: list[int], one_class: bool, classes: Sequence[str] | None) -> dict[str, Any]:
    if one_class:
        name = classes[present[0]] if classes and present[0] < len(classes) else present[0]
        return _missing(
            f"the test split holds a single class ({name}), so ROC-AUC is undefined"
        )
    if proba is None:
        return _missing("this Model does not output class probabilities")
    try:
        if len(present) == 2:
            positive = present[1] if present[0] == 0 else present[0]
            if proba.shape[1] <= max(present):
                return _missing("the probabilities do not cover every class in the test split")
            return _measured(skm.roc_auc_score(truth, proba[:, positive]))
        if proba.shape[1] < len(present):
            return _missing("the probabilities do not cover every class in the test split")
        return _measured(
            skm.roc_auc_score(
                truth, proba[:, present], multi_class="ovr", average="macro", labels=present
            )
        )
    except ValueError as exc:
        return _missing(f"ROC-AUC could not be computed: {exc}")


def evaluate_regression(y_true: Sequence[Any], y_pred: Sequence[Any]) -> dict[str, Any]:
    """Regression metrics on the held-out test split."""
    from sklearn import metrics as skm  # optional extra, imported lazily

    truth = np.asarray(y_true, dtype="float64")
    predicted = np.asarray(y_pred, dtype="float64")
    out: dict[str, Any] = {
        "mae": _measured(skm.mean_absolute_error(truth, predicted)),
        "rmse": _measured(skm.mean_squared_error(truth, predicted) ** 0.5),
        "median_absolute_error": _measured(skm.median_absolute_error(truth, predicted)),
        "explained_variance": _measured(skm.explained_variance_score(truth, predicted)),
        "max_error": _measured(skm.max_error(truth, predicted)),
    }
    if len(truth) < 2:
        out["r2"] = _missing("R² needs at least 2 held-out rows")
    else:
        out["r2"] = _measured(skm.r2_score(truth, predicted))
    out["n_test"] = len(truth)
    out["residual_mean"] = _measured(float(np.mean(truth - predicted)))
    return out


# -- the leaderboard ---------------------------------------------------------


def rank_leaderboard(entries: Sequence[Mapping[str, Any]], primary_metric: str) -> list[dict[str, Any]]:
    """Attach a rank by the primary metric: best first, unmeasurable last.

    Ties keep the registry's order so two identical Models are not re-ordered
    between runs. A failed Model always ranks last, whatever it scored.
    """
    spec = METRICS[primary_metric]
    rows = [dict(entry) for entry in entries]

    def sort_key(entry: dict[str, Any]) -> tuple[int, float, int]:
        if entry.get("status") != "ok":
            return (2, 0.0, 0)
        value = metric_value(entry.get("metrics") or {}, primary_metric)
        if value is None:
            return (1, 0.0, 0)
        return (0, -value if spec.higher_is_better else value, 0)

    rows.sort(key=sort_key)
    ranked: list[dict[str, Any]] = []
    previous_value: float | None = None
    previous_rank = 0
    for position, entry in enumerate(rows):
        value = metric_value(entry.get("metrics") or {}, primary_metric)
        comparable = entry.get("status") == "ok" and value is not None
        if comparable and previous_value is not None and value == previous_value:
            entry["rank"] = previous_rank
        else:
            entry["rank"] = position + 1 if comparable else None
            if comparable:
                previous_rank = entry["rank"]
        previous_value = value if comparable else None
        entry["primary_metric"] = primary_metric
        entry["primary"] = (entry.get("metrics") or {}).get(primary_metric)
        ranked.append(entry)
    return ranked


def leaderboard_warnings(
    entries: Sequence[Mapping[str, Any]], primary_metric: str
) -> list[str]:
    """Honest notes about a leaderboard that could not rank everything."""
    warnings: list[str] = []
    fitted = [entry for entry in entries if entry.get("status") == "ok"]
    ranked = [e for e in fitted if metric_value(e.get("metrics") or {}, primary_metric) is not None]
    if not ranked:
        warnings.append(
            f"no Model could be scored on {primary_metric!r}, so nothing was ranked"
        )
    unranked = [e for e in fitted if e not in ranked]
    if unranked:
        reasons = sorted(
            {
                str((e.get("metrics") or {}).get(primary_metric, {}).get("reason") or "unknown")
                for e in unranked
            }
        )
        warnings.append(
            f"{len(unranked)} Model(s) could not be scored on {primary_metric!r} and are "
            f"listed without a rank: {'; '.join(reasons)}"
        )
    failed = [e for e in entries if e.get("status") != "ok"]
    if failed:
        warnings.append(
            f"{len(failed)} Model(s) failed to fit and are listed last: "
            + ", ".join(str(e.get("model")) for e in failed)
        )
    return warnings


# -- tuning ------------------------------------------------------------------


def _will_have_proba(spec: ModelSpec, hyperparameters: Mapping[str, Any]) -> bool:
    """Whether this configuration can produce class probabilities.

    The registry flag describes the *default*; an SVM the user asked to calibrate
    (``probability: true``) can, even though the flag says otherwise.
    """
    if spec.supports_proba:
        return True
    return bool(hyperparameters.get("probability"))


def _search_space(
    spec: ModelSpec, overrides: Mapping[str, Any] | None, custom: Mapping[str, Sequence[Any]] | None
) -> dict[str, list[Any]]:
    space: dict[str, list[Any]] = {}
    for name, values in (custom or {}).items():
        if not values:
            raise ModelError(f"search space for {name!r} is empty")
        space[str(name)] = list(values)
    for name, values in spec.search_space.items():
        if (overrides or {}) and name in (overrides or {}):
            continue  # a pinned hyperparameter is not searched
        space.setdefault(name, list(values))
    if not space:
        raise ModelError(
            f"Model {spec.name!r} has no hyperparameter to search; turn tuning off for it "
            "or pin its hyperparameters"
        )
    return space


def _cv_splitter(task_type: str, cv: int, seed: int) -> Any:
    from sklearn.model_selection import KFold, StratifiedKFold  # optional extra

    folds = max(2, int(cv))
    if task_type == CLASSIFICATION:
        return StratifiedKFold(n_splits=folds, shuffle=True, random_state=seed)
    return KFold(n_splits=folds, shuffle=True, random_state=seed)


def _aligned_proba(estimator: Any, matrix: np.ndarray) -> np.ndarray | None:
    """``predict_proba`` columns reindexed onto the encoded class codes."""
    if not hasattr(estimator, "predict_proba"):
        return None
    raw = np.asarray(estimator.predict_proba(matrix), dtype="float64")
    own = [int(value) for value in getattr(estimator, "classes_", [])]
    if not own or own == list(range(len(own))):
        return raw
    out = np.zeros_like(raw)
    for column, code in enumerate(own):
        if 0 <= code < raw.shape[1]:
            out[:, code] = raw[:, column]
    return out


def _adaptive_roc_auc(estimator: Any, matrix: np.ndarray, truth: np.ndarray) -> float:
    """ROC-AUC that works for a binary fold and a multiclass fold alike.

    scikit-learn's own ``roc_auc`` scorer refuses a multiclass Target unless
    ``multi_class`` is set, and a fold does not declare its Task Type — a
    3-class Target has binary folds as soon as one class is rare. This picks the
    right form from the fold it is scoring, and returns the neutral 0.5 for a
    fold that cannot discriminate.
    """
    from sklearn.metrics import roc_auc_score  # optional extra

    proba = _aligned_proba(estimator, matrix)
    present = sorted(set(int(value) for value in truth))
    if proba is None or len(present) < 2 or proba.shape[1] <= max(present):
        return 0.5
    if len(present) == 2:
        positive = present[1] if present[0] == 0 else present[0]
        return float(roc_auc_score(truth, proba[:, positive]))
    return float(
        roc_auc_score(
            truth, proba[:, present], multi_class="ovr", average="macro", labels=present
        )
    )


def _min_class_f1(estimator: Any, matrix: np.ndarray, truth: np.ndarray) -> float:
    """F1 of the least frequent class in the fold."""
    from sklearn.metrics import f1_score  # optional extra

    predicted = np.asarray(estimator.predict(matrix))
    present = sorted(set(int(value) for value in truth))
    if len(present) < 2:
        return 0.0
    smallest = min(present, key=lambda code: int((truth == code).sum()))
    return float(
        f1_score(truth, predicted, labels=[smallest], average="macro", zero_division=0)
    )


class _FoldScorer:
    """A ``(estimator, X, y) -> float`` callable for a search.

    ``make_scorer`` would call a ``(y_true, y_pred)`` function, which is no use
    here: these two metrics need the estimator itself. scikit-learn accepts any
    callable with this signature as ``scoring``.
    """

    name = "metric"

    def __call__(self, estimator: Any, matrix: Any, truth: Any) -> float:  # pragma: no cover
        raise NotImplementedError


class _RocAucScorer(_FoldScorer):
    name = "roc_auc"

    def __call__(self, estimator: Any, matrix: Any, truth: Any) -> float:
        return _adaptive_roc_auc(estimator, matrix, np.asarray(truth))


class _MinClassF1Scorer(_FoldScorer):
    name = "f1_minority"

    def __call__(self, estimator: Any, matrix: Any, truth: Any) -> float:
        return _min_class_f1(estimator, matrix, np.asarray(truth))


def _scorer_for(metric: str) -> Any:
    """The search scorer behind one of our metrics.

    Most map onto a named scorer (the error metrics are negated by scikit-learn,
    which is why MAE and RMSE rank correctly). ROC-AUC and the smallest-class F1
    get a callable of our own, because a fold does not declare its Task Type.
    """
    from sklearn.metrics import get_scorer  # optional extra

    spec = METRICS[metric]
    if metric == "roc_auc":
        return _RocAucScorer()
    if metric == "f1_minority":
        return _MinClassF1Scorer()
    if spec.scorer:
        return get_scorer(spec.scorer)
    raise ModelError(f"metric {metric!r} cannot drive a search")


def _space_size(space: Mapping[str, Sequence[Any]]) -> int:
    total = 1
    for values in space.values():
        total *= max(1, len(values))
    return total


def tune_model(
    spec: ModelSpec,
    data: TrainingMatrix,
    *,
    metric: str,
    strategy: str = "random",
    n_iter: int = 10,
    cv: int = 3,
    search_space: Mapping[str, Sequence[Any]] | None = None,
    overrides: Mapping[str, Any] | None = None,
    seed: int = 0,
) -> dict[str, Any]:
    """Search ``data`` — the training matrix — for better hyperparameters.

    The signature is the guarantee: the search only accepts a
    :class:`TrainingMatrix`, and the only way to build one is
    :meth:`TrainingMatrix.from_split`, which reads ``split.train``. There is no
    parameter through which the held-out test rows could be passed.

    Returns a plain dict for the Training Run: the winning hyperparameters, the
    cross-validated score, and exactly which rows were searched.
    """
    if not isinstance(data, TrainingMatrix):  # pragma: no cover - defensive
        raise TypeError("tuning only accepts a TrainingMatrix (the training split)")
    metric_spec = METRICS[metric]
    if metric_spec.task_type != data.task_type:
        raise ModelError(
            f"metric {metric!r} is a {metric_spec.task_type} metric but this Training Run "
            f"is a {data.task_type} Target"
        )
    base = resolve_hyperparameters(spec, data.task_type, seed=seed, overrides=overrides)
    if metric_spec.needs_proba and not _will_have_proba(spec, base):
        return {
            "model": spec.name,
            "skipped": f"Model {spec.name!r} does not output class probabilities, so it "
            f"cannot be tuned on {metric!r}",
            "trained_on_rows": list(data.rows),
            "n_samples": data.n_samples,
        }
    if data.n_samples < 2 * max(2, int(cv)):
        return {
            "model": spec.name,
            "skipped": f"only {data.n_samples} training rows, too few for {cv}-fold "
            "cross-validation",
            "trained_on_rows": list(data.rows),
            "n_samples": data.n_samples,
        }
    from sklearn.model_selection import (  # optional extra
        GridSearchCV,
        RandomizedSearchCV,
    )

    space = _search_space(spec, overrides, search_space)
    estimator = build_estimator(spec, data.task_type, hyperparameters=base)
    scorer = _scorer_for(metric)
    splitter = _cv_splitter(data.task_type, cv, seed)
    searcher = (
        GridSearchCV(estimator, space, cv=splitter, scoring=scorer)
        if strategy == "grid"
        else RandomizedSearchCV(
            estimator,
            space,
            n_iter=max(1, min(int(n_iter), _space_size(space))),
            cv=splitter,
            scoring=scorer,
            random_state=seed,
            n_jobs=1,
        )
    )
    searcher.fit(data.X, data.y)
    score = float(searcher.best_score_)
    searched = {
        "model": spec.name,
        "strategy": strategy,
        "metric": metric,
        "cv": max(2, int(cv)),
        "candidates": len(searcher.cv_results_["params"]),
        "search_space": {
            name: [_jsonable(value) for value in values] for name, values in space.items()
        },
        "trained_on_rows": list(data.rows),
        "n_samples": data.n_samples,
    }
    if not np.isfinite(score):
        # Every candidate failed to score: say so rather than reporting a winner
        # picked out of a set of NaNs.
        return {
            **searched,
            "skipped": f"no candidate could be cross-validated on {metric!r}",
        }
    return {
        **searched,
        "skipped": None,
        "best_params": {
            name: _jsonable(value) for name, value in searcher.best_params_.items()
        },
        #: The whole mapping the winning estimator was built with, so a pinned
        #: hyperparameter and the seed show up next to the searched winners.
        "best_hyperparameters": {
            name: _jsonable(value)
            for name, value in resolve_hyperparameters(
                spec,
                data.task_type,
                seed=seed,
                overrides={**dict(overrides or {}), **searcher.best_params_},
            ).items()
        },
        "best_score": round(score, 6),
    }


def _jsonable(value: Any) -> Any:
    """A searched hyperparameter as plain JSON (numpy scalars included)."""
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, np.ndarray):
        return [_jsonable(v) for v in value.tolist()]
    return value


__all__ = [
    "CLASSIFICATION",
    "METRICS",
    "MODEL_NAMES",
    "MODEL_SPECS",
    "REGRESSION",
    "TASK_TYPES",
    "FittedModel",
    "MetricSpec",
    "ModelError",
    "ModelSpec",
    "ModelUnavailableError",
    "TrainingMatrix",
    "UnsupportedModelError",
    "assert_held_out",
    "available_models",
    "build_estimator",
    "default_primary_metric",
    "encode_classes",
    "encode_regression",
    "evaluate_classification",
    "evaluate_regression",
    "fit_model",
    "get_spec",
    "is_available",
    "leaderboard_warnings",
    "library_versions",
    "metric_value",
    "metrics_for",
    "rank_leaderboard",
    "resolve_hyperparameters",
    "resolve_models",
    "resolve_primary_metric",
    "tune_model",
    "value_key",
]
