"""The MLP **Models**: one dense network, trained in PyTorch or TensorFlow.

The two frameworks are optional extras and neither is needed to *import* this
module: :class:`TorchMLP` and :class:`TensorFlowMLP` are ordinary classes whose
``__init__`` touches nothing but their own arguments, and the framework is
imported inside :meth:`~MlpModel.fit`. A bare install therefore still gets the
registry, the plan, the leaderboard shape, and a readable "not installed"
refusal naming the extra to install.

Why an estimator shape at all
-----------------------------
A **Model** on the leaderboard is something with ``fit``, ``predict`` and —
for a classification Target — ``predict_proba``. Wrapping each framework in that
shape is what lets an MLP share the *whole* of the existing path:
:func:`dat_distiller.training.trainers.build_estimator` instantiates it,
:func:`~dat_distiller.training.trainers.fit_model` times it, the metrics are the
same functions the scikit-learn Models are scored with, and
:func:`~dat_distiller.training.trainers.tune_model` can cross-validate it
because it is cloneable. There is deliberately no second scoring
implementation anywhere in this module: "comparable metrics" has to mean the
same metric *names* computed by the same code, not a parallel set.

The validation signal for early stopping
----------------------------------------
Early stopping needs a validation signal, and the held-out test split is off
limits: it is the one split a Training Run may touch exactly once, to score.
So early stopping carves a ``validation_fraction`` slice (default 0.15,
stratified by Target class for a classification) out of whatever ``fit`` was
handed — and ``fit`` is only ever handed the **training split**, by
``trainers.fit_model``. The slice therefore comes from the training rows by
construction, not by discipline: the held-out rows are not in the array the
network ever sees. ``training_report_`` records how many rows went to each
side and why the loop stopped.

Determinism
-----------
Both networks are seeded from the Training Run's seed, run single-threaded and
restore their best-scoring weights at the end of the loop, so the same seed and
the same rows produce the same numbers and ``predict`` does not depend on when
the loop happened to stop. The per-framework knobs
(:func:`_torch_deterministic`, :func:`_tf_deterministic`) are best-effort on
purpose: a framework that refuses a determinism setting is trained anyway, with
the reason recorded in ``training_report_``, because a Training Run failing over
a determinism flag would be a worse outcome than a possibly non-reproducible
one.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from importlib import import_module
from typing import Any

import numpy as np

from .trainers import CLASSIFICATION, TASK_TYPES, ModelError

#: The default fraction of the *training* rows early stopping holds back to watch.
#: The held-out test split is never used for this; see the module docstring.
DEFAULT_VALIDATION_FRACTION = 0.15


# -- the options, framework-free ---------------------------------------------


@dataclass(frozen=True)
class MlpOptions:
    """One MLP configuration, validated once and shared by both frameworks."""

    task_type: str
    hidden_units: tuple[int, ...]
    epochs: int
    learning_rate: float
    batch_size: int
    dropout: float
    early_stopping: bool
    patience: int
    validation_fraction: float
    seed: int

    @property
    def is_classification(self) -> bool:
        return self.task_type == CLASSIFICATION

    def to_dict(self) -> dict[str, Any]:
        return {
            "task_type": self.task_type,
            "hidden_units": list(self.hidden_units),
            "epochs": self.epochs,
            "learning_rate": self.learning_rate,
            "batch_size": self.batch_size,
            "dropout": self.dropout,
            "early_stopping": self.early_stopping,
            "patience": self.patience,
            "validation_fraction": self.validation_fraction,
            "seed": self.seed,
        }


def _number(value: Any, name: str) -> float:
    """A hyperparameter as a float, refusing a non-number readably.

    ``float("fast")`` raises a bare ``ValueError`` that names the value and not
    the hyperparameter, which is a poor thing to show a user who mistyped a
    learning rate.
    """
    if isinstance(value, bool) or not isinstance(value, (int, float, np.integer, np.floating)):
        raise ModelError(f"MLP hyperparameter {name!r} must be a number, got {value!r}")
    return float(value)


def _positive_int(value: Any, name: str, minimum: int = 1) -> int:
    number = _number(value, name)
    if not float(number).is_integer():
        raise ModelError(f"MLP hyperparameter {name!r} must be a whole number, got {value!r}")
    if number < minimum:
        raise ModelError(f"MLP hyperparameter {name!r} must be at least {minimum}, got {number:g}")
    return int(number)


def resolve_mlp_options(hyperparameters: Mapping[str, Any], task_type: str) -> MlpOptions:
    """Validate one MLP's hyperparameters into a shared :class:`MlpOptions`.

    Raising :class:`~dat_distiller.training.trainers.ModelError` (a
    ``ValueError``) is what a bad override should do: the Training Run records
    the Model as failed with this message and the rest of the leaderboard still
    ranks, exactly as it does for a scikit-learn Model given a nonsense value.
    """
    if task_type not in TASK_TYPES:
        raise ModelError(f"task type must be one of {TASK_TYPES}, got {task_type!r}")

    raw_units = hyperparameters.get("hidden_units", (64, 32))
    if isinstance(raw_units, (str, bytes)) or not isinstance(raw_units, Sequence):
        raise ModelError(
            f"MLP hyperparameter 'hidden_units' must be a list of layer widths, got {raw_units!r}"
        )
    hidden_units = tuple(_positive_int(unit, f"hidden_units[{i}]") for i, unit in enumerate(raw_units))
    if not hidden_units:
        raise ModelError(
            "MLP hyperparameter 'hidden_units' must name at least one hidden layer; "
            "use a linear regression Model for a linear model"
        )

    learning_rate = _number(hyperparameters.get("learning_rate", 1e-3), "learning_rate")
    if not np.isfinite(learning_rate) or learning_rate <= 0.0:
        raise ModelError(
            f"MLP hyperparameter 'learning_rate' must be a positive number, got {learning_rate!r}"
        )

    dropout = _number(hyperparameters.get("dropout", 0.0), "dropout")
    if not np.isfinite(dropout) or not 0.0 <= dropout < 1.0:
        raise ModelError(
            f"MLP hyperparameter 'dropout' must be at least 0 and below 1, got {dropout!r}"
        )

    validation_fraction = _number(
        hyperparameters.get("validation_fraction", DEFAULT_VALIDATION_FRACTION),
        "validation_fraction",
    )
    if not np.isfinite(validation_fraction) or not 0.0 < validation_fraction < 1.0:
        raise ModelError(
            "MLP hyperparameter 'validation_fraction' must be between 0 and 1 (exclusive), "
            f"got {validation_fraction!r}"
        )

    return MlpOptions(
        task_type=task_type,
        hidden_units=hidden_units,
        epochs=_positive_int(hyperparameters.get("epochs", 50), "epochs"),
        learning_rate=learning_rate,
        batch_size=_positive_int(hyperparameters.get("batch_size", 32), "batch_size"),
        dropout=dropout,
        early_stopping=bool(hyperparameters.get("early_stopping", True)),
        patience=_positive_int(hyperparameters.get("patience", 10), "patience"),
        validation_fraction=validation_fraction,
        seed=_positive_int(hyperparameters.get("seed", 0), "seed", minimum=0),
    )


def validation_indices(
    values: np.ndarray, fraction: float, seed: int, *, stratified: bool
) -> np.ndarray:
    """The row positions held back to watch, as a deterministic function of the seed.

    Stratified for a classification so every class keeps a row on both sides
    where it has one to spare; a regression Target is one group. At least one
    row stays behind for fitting and at least one stays in the training part.
    """
    rng = np.random.default_rng(seed)
    rows = np.arange(len(values))
    groups = [np.flatnonzero(values == code) for code in np.unique(values)] if stratified else [rows]
    taken: list[np.ndarray] = []
    for group in groups:
        if len(group) < 2:
            continue
        count = round(len(group) * fraction)
        count = min(max(count, 1), len(group) - 1)
        taken.append(rng.choice(group, size=count, replace=False))
    if not taken:
        return np.empty(0, dtype="int64")
    return np.sort(np.concatenate(taken)).astype("int64")


# -- the estimator shape -----------------------------------------------------


class MlpModel:
    """What every framework's wrapper must be: cloneable, and fittable.

    ``get_params``/``set_params`` are written out rather than inherited from
    scikit-learn's ``BaseEstimator`` so that this module does not need the
    ``sklearn`` extra: a torch-only install can still train a Model. They
    follow the same rule sklearn's clone relies on — ``__init__`` stores exactly
    what it was given, unconverted, and every argument is readable by name.
    """

    #: Read by :func:`dat_distiller.training.trainers.build_estimator`, which
    #: injects the Target's Task Type. The two heads (softmax over the classes,
    #: a single linear unit) are different networks, so the Task Type has to
    #: reach the constructor; guessing it from the Target's dtype would break
    #: the moment a classification Target held whole numbers.
    wants_task_type = True

    #: The distribution this class needs, for the "not installed" message.
    extra = ""

    def __init__(
        self,
        *,
        task_type: str = CLASSIFICATION,
        hidden_units: Sequence[int] = (64, 32),
        epochs: int = 50,
        learning_rate: float = 1e-3,
        batch_size: int = 32,
        dropout: float = 0.0,
        early_stopping: bool = True,
        patience: int = 10,
        validation_fraction: float = DEFAULT_VALIDATION_FRACTION,
        seed: int = 0,
    ) -> None:
        self.task_type = task_type
        self.hidden_units = hidden_units
        self.epochs = epochs
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.dropout = dropout
        self.early_stopping = early_stopping
        self.patience = patience
        self.validation_fraction = validation_fraction
        self.seed = seed

    # -- the sklearn estimator protocol, minus the sklearn dependency ---------

    #: The constructor's arguments, in order. ``get_params``/``set_params`` and
    #: scikit-learn's ``clone`` both read this, and every one of them is stored
    #: verbatim under its own name so the mapping round-trips.
    PARAMS = (
        "task_type",
        "hidden_units",
        "epochs",
        "learning_rate",
        "batch_size",
        "dropout",
        "early_stopping",
        "patience",
        "validation_fraction",
        "seed",
    )

    @property
    def is_classification(self) -> bool:
        return self.task_type == CLASSIFICATION

    def get_params(self, deep: bool = True) -> dict[str, Any]:
        return {name: getattr(self, name) for name in self.PARAMS}

    def set_params(self, **params: Any) -> MlpModel:
        for name, value in params.items():
            if name not in self.get_params():
                raise ValueError(f"invalid hyperparameter {name!r} for {type(self).__name__}")
            setattr(self, name, value)
        return self

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return f"{type(self).__name__}({self.get_params()})"

    # -- the tags a cross-validation search asks for -------------------------

    def __sklearn_tags__(self) -> Any:
        """scikit-learn >= 1.6 asks every estimator for its tags before cloning.

        Written out rather than inherited from ``BaseEstimator`` so this module
        keeps working without the ``sklearn`` extra. The old dict-based protocol
        is answered too, so the same class cross-validates on scikit-learn 1.4
        and 1.5, which the ``sklearn`` extra still allows.
        """
        try:
            from sklearn.utils import Tags, TargetTags
        except ImportError:  # pragma: no cover - scikit-learn < 1.6
            return {"requires_fit": True, "non_deterministic": False}
        return Tags(
            estimator_type=None,
            target_tags=TargetTags(required=True),
            regressor_tags=None if self.is_classification else _regressor_tags(),
            classifier_tags=_classifier_tags() if self.is_classification else None,
        )

    def _more_tags(self) -> dict[str, Any]:  # pragma: no cover - scikit-learn < 1.6
        """The dict tag protocol scikit-learn 1.4 and 1.5 use."""
        return {"requires_fit": True, "non_deterministic": False}

    # -- what the frameworks implement ---------------------------------------

    def fit(self, matrix: Any, values: Any) -> MlpModel:
        raise NotImplementedError

    def predict(self, matrix: Any) -> np.ndarray:
        raise NotImplementedError

    @property
    def predict_proba(self) -> Any:
        """Class probabilities — on a classification Model only.

        A *property* rather than a method, because that is how the rest of the
        codebase asks the question: :meth:`FittedModel.proba` does
        ``hasattr(estimator, "predict_proba")``, and a regression Target has no
        class probabilities to report. Raising ``AttributeError`` here is what
        makes the answer ``None`` — a null ROC-AUC *with a reason* — instead of a
        nonsense array. It is the same distinction scikit-learn draws between
        ``SVC`` and ``SVR``.
        """
        if not self.is_classification:
            raise AttributeError(
                f"{type(self).__name__} is fitted on a regression Target and has no "
                "class probabilities"
            )
        return self._predict_proba

    def _predict_proba(self, matrix: Any) -> np.ndarray:
        raise NotImplementedError


def _classifier_tags() -> Any:
    from sklearn.utils import ClassifierTags

    return ClassifierTags()


def _regressor_tags() -> Any:
    from sklearn.utils import RegressorTags

    return RegressorTags()


def _report(
    options: MlpOptions,
    *,
    n_train: int,
    n_validation: int,
    n_features: int,
    epochs_run: int,
    best_epoch: int,
    best_validation_loss: float | None,
    stop_reason: str,
    determinism: Mapping[str, str],
) -> dict[str, Any]:
    """What the Training Run would want to know about how the network was fitted.

    Read by the Model Bundle (#34). It is deliberately *not* merged into the
    leaderboard rows: every Model on the board has the same shape, and that is
    the thing the issue is about.
    """
    return {
        "architecture": {
            "input_units": n_features,
            "hidden_units": list(options.hidden_units),
            "output_units": "softmax" if options.is_classification else 1,
            "dropout": options.dropout,
        },
        "optimisation": {
            "loss": "cross_entropy" if options.is_classification else "mean_squared_error",
            "optimizer": "adam",
            "learning_rate": options.learning_rate,
            "batch_size": options.batch_size,
            "seed": options.seed,
        },
        "early_stopping": {
            "enabled": options.early_stopping,
            "patience": options.patience,
            "validation_fraction": options.validation_fraction if options.early_stopping else 0.0,
            "validation_rows": n_validation,
            "training_rows": n_train,
            "validation_split_from": "the training split, never the held-out test split",
        },
        "epochs": {
            "configured": options.epochs,
            "run": epochs_run,
            "best": best_epoch,
            "stopped_because": stop_reason,
            # `None`, never `nan`: this report is destined for a checkpoint, and
            # `nan` is not valid JSON. No validation slice means no best epoch.
            "best_validation_loss": (
                None
                if best_validation_loss is None or not np.isfinite(best_validation_loss)
                else round(float(best_validation_loss), 6)
            ),
        },
        "determinism": dict(determinism),
    }


# -- PyTorch -----------------------------------------------------------------


def _torch_deterministic(seed: int) -> dict[str, str]:
    """Seed PyTorch and ask for deterministic kernels, reporting what it granted.

    The CPU kernels this Model uses (dense layers, ReLU, dropout, Adam) are all
    deterministic; single-threading removes the only remaining source of drift,
    the order float reductions are summed in. ``use_deterministic_algorithms``
    is asked for with ``warn_only`` because an op this Model never reaches
    should not be able to fail a Training Run.
    """
    torch = import_module("torch")
    notes: dict[str, str] = {}
    try:
        # One thread: the CPU kernels this Model uses are individually
        # deterministic, but their float reductions are summed in whatever order
        # the thread pool hands them out. Fewer threads also means a faster fit
        # of a net this small.
        torch.set_num_threads(1)
        notes["threads"] = "1"
    except (RuntimeError, ValueError) as exc:  # pragma: no cover - platform dependent
        notes["threads"] = f"unchanged ({exc})"
    try:
        torch.use_deterministic_algorithms(True, warn_only=True)
        notes["deterministic_algorithms"] = "requested"
    except (AttributeError, RuntimeError, ValueError) as exc:  # pragma: no cover - version dependent
        notes["deterministic_algorithms"] = f"unavailable ({exc})"
    torch.manual_seed(int(seed))
    notes["seeded"] = f"torch.manual_seed({int(seed)})"
    return notes


class TorchMLP(MlpModel):
    """A dense MLP fitted with PyTorch.

    Parameters
    ----------
    hidden_units
        Width of each hidden layer, in order; ``[64, 32]`` is two hidden layers.
    epochs
        The cap on passes over the training rows. With ``early_stopping`` the
        loop usually stops sooner and the best-scoring weights are restored.
    learning_rate, batch_size, dropout
        Adam's step size, the mini-batch size, and the dropout rate applied
        after each hidden layer.
    early_stopping, patience, validation_fraction
        Stop after ``patience`` epochs without an improvement in the validation
        loss, watched on a ``validation_fraction`` slice of the **training**
        rows (see the module docstring). ``early_stopping=False`` trains for the
        full ``epochs`` and ignores the fraction.
    seed
        Seeds weight initialisation, the dropout masks and the mini-batch
        order, so two runs of the same Training Run agree.
    """

    extra = "torch"

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)

    def _module(self) -> Any:
        try:
            return import_module("torch")
        except ImportError as exc:  # pragma: no cover - guarded by the registry
            raise ImportError(
                "PyTorch is not installed; install the optional extra with "
                "`uv sync --extra torch` or `uv pip install torch`"
            ) from exc

    # -- building ---------------------------------------------------------

    def _build(self, torch: Any, nn: Any, n_features: int, n_outputs: int) -> Any:
        layers: list[Any] = []
        width = n_features
        for units in self.hidden_units:
            layers.append(nn.Linear(width, units))
            layers.append(nn.ReLU())
            if self.dropout:
                layers.append(nn.Dropout(float(self.dropout)))
            width = units
        layers.append(nn.Linear(width, n_outputs))
        return nn.Sequential(*layers)

    def _loss(self, nn: Any) -> Any:
        return nn.CrossEntropyLoss() if self.is_classification else nn.MSELoss()

    # -- fitting ----------------------------------------------------------

    def fit(self, matrix: Any, values: Any) -> TorchMLP:
        torch = self._module()
        nn = import_module("torch.nn")
        options = resolve_mlp_options(self.get_params(), self.task_type)
        determinism = _torch_deterministic(options.seed)

        X = np.ascontiguousarray(matrix, dtype="float32")
        y = np.asarray(values)
        codes, targets = _targets(y, options)
        n_classes = int(codes.max()) + 1 if options.is_classification else 1

        keep = self._training_positions(codes, options)
        X_fit, y_fit = X[keep], targets[keep]
        watch = np.setdiff1d(self._watch_positions(codes, options), keep)
        X_watch = X[watch] if watch.size else None

        model = self._build(torch, nn, X.shape[1], n_classes)
        optimizer = torch.optim.Adam(model.parameters(), lr=options.learning_rate)
        loss_fn = self._loss(nn)
        generator = np.random.default_rng(options.seed)

        best_loss = np.inf
        best_state: dict[str, Any] | None = None
        best_epoch = 0
        since_improvement = 0
        epochs_run = 0
        stop_reason = "reached the configured number of epochs"
        for epoch in range(options.epochs):
            epochs_run = epoch + 1
            model.train()
            order = generator.permutation(len(X_fit))
            for start in range(0, len(X_fit), options.batch_size):
                batch = torch.from_numpy(X_fit[order[start : start + options.batch_size]])
                loss = loss_fn(
                    self._forward_logits(torch, model, batch),
                    torch.from_numpy(y_fit[order[start : start + options.batch_size]]),
                )
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            if X_watch is None:
                continue
            watched = self._validation_loss(torch, model, loss_fn, X_watch, targets[watch])
            if watched < best_loss:
                best_loss, best_epoch, since_improvement = watched, epoch + 1, 0
                best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
            else:
                since_improvement += 1
                if since_improvement >= options.patience:
                    stop_reason = (
                        f"the validation loss did not improve for {options.patience} epochs"
                    )
                    break
        if best_state is not None:
            # Restore the best weights, so predict() does not depend on when the
            # loop stopped.
            model.load_state_dict(best_state)

        self._model = model
        self._n_classes = n_classes
        self.classes_ = np.unique(codes)
        self.training_report_ = _report(
            options,
            n_train=len(keep),
            n_validation=watch.size,
            n_features=int(X.shape[1]),
            epochs_run=epochs_run,
            best_epoch=best_epoch,
            best_validation_loss=best_loss,
            stop_reason=stop_reason,
            determinism={"torch": determinism},
        )
        return self

    # -- predicting -------------------------------------------------------

    def predict(self, matrix: Any) -> np.ndarray:
        torch = self._module()
        logits = self._logits(torch, matrix)
        if self.is_classification:
            return np.asarray(logits.argmax(axis=1), dtype="int64")
        return np.asarray(logits[:, 0], dtype="float64")

    def _predict_proba(self, matrix: Any) -> np.ndarray:
        torch = self._module()
        return np.asarray(torch.softmax(self._logits(torch, matrix), dim=1), dtype="float64")

    # -- internals --------------------------------------------------------

    def _training_positions(self, codes: np.ndarray, options: MlpOptions) -> np.ndarray:
        if not options.early_stopping:
            return np.arange(len(codes), dtype="int64")
        watch = self._watch_positions(codes, options)
        if watch.size == 0:
            return np.arange(len(codes), dtype="int64")
        return np.setdiff1d(np.arange(len(codes), dtype="int64"), watch)

    def _watch_positions(self, codes: np.ndarray, options: MlpOptions) -> np.ndarray:
        if not options.early_stopping:
            return np.empty(0, dtype="int64")
        return validation_indices(
            codes, options.validation_fraction, options.seed, stratified=options.is_classification
        )

    def _forward_logits(self, torch: Any, model: Any, batch: Any) -> Any:
        if self.is_classification:
            return model(batch)
        return model(batch)[:, 0]

    def _validation_loss(self, torch: Any, model: Any, loss_fn: Any, X: Any, y: Any) -> float:
        if len(X) == 0:
            return float("nan")
        model.eval()
        with torch.no_grad():
            logits = self._forward_logits(torch, model, torch.from_numpy(X))
            return float(loss_fn(logits, torch.from_numpy(y)).item())

    def _logits(self, torch: Any, matrix: Any) -> Any:
        if not hasattr(self, "_model"):
            raise ModelError(f"{type(self).__name__} has not been fitted yet")
        self._model.eval()
        with torch.no_grad():
            return self._model(torch.from_numpy(np.ascontiguousarray(matrix, dtype="float32")))


# -- TensorFlow --------------------------------------------------------------


def _tf_deterministic(seed: int) -> dict[str, str]:
    """Seed Keras and ask TensorFlow for deterministic ops, reporting what it granted.

    ``enable_op_determinism`` rewrites stateful ops (the shuffling Keras does
    between batches is one) to read from seeded counters instead of a global
    counter, which is what makes a repeated fit of the same rows agree. It is
    process-wide and can refuse on some builds, so a refusal is recorded rather
    than raised.
    """
    tf = import_module("tensorflow")
    notes: dict[str, str] = {}
    try:
        tf.keras.utils.set_random_seed(int(seed))
        notes["seeded"] = f"tf.keras.utils.set_random_seed({int(seed)})"
    except (AttributeError, RuntimeError, ValueError) as exc:  # pragma: no cover - version dependent
        notes["seeded"] = f"tf.random.set_seed only ({exc})"
        tf.random.set_seed(int(seed))
    try:
        tf.config.experimental.enable_op_determinism()
        notes["op_determinism"] = "enabled"
    except (AttributeError, RuntimeError, ValueError) as exc:  # pragma: no cover - build dependent
        notes["op_determinism"] = f"unavailable ({exc})"
    return notes


class TensorFlowMLP(MlpModel):
    """A dense MLP fitted with TensorFlow/Keras.

    The parameters mean exactly what they mean on :class:`TorchMLP`; see its
    docstring. The two are the same **Model** in two libraries, which is what
    makes them comparable on one leaderboard.
    """

    extra = "tensorflow"

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)

    def _module(self) -> Any:
        try:
            return import_module("tensorflow")
        except ImportError as exc:  # pragma: no cover - guarded by the registry
            raise ImportError(
                "TensorFlow is not installed; install the optional extra with "
                "`uv sync --extra tensorflow` or `uv pip install tensorflow`"
            ) from exc

    # -- building ---------------------------------------------------------

    def _build(self, tf: Any, n_features: int, n_outputs: int, options: MlpOptions) -> Any:
        # One Input, then the hidden widths from `options` (not from `self`, so
        # it is the validated ones that actually get built).
        layers: list[Any] = [tf.keras.layers.Input(shape=(n_features,))]
        for units in options.hidden_units:
            layers.append(tf.keras.layers.Dense(int(units), activation="relu"))
            if options.dropout:
                layers.append(tf.keras.layers.Dropout(options.dropout))
            n_features = int(units)
        layers.append(tf.keras.layers.Dense(n_outputs))
        return tf.keras.Sequential(layers)

    def _loss(self) -> Any:
        return (
            tf_loss_sparse()
            if self.is_classification
            else import_module("tensorflow").keras.losses.MeanSquaredError()
        )

    # -- fitting ----------------------------------------------------------

    def fit(self, matrix: Any, values: Any) -> TensorFlowMLP:
        tf = self._module()
        options = resolve_mlp_options(self.get_params(), self.task_type)
        determinism = _tf_deterministic(options.seed)

        X = np.ascontiguousarray(matrix, dtype="float32")
        y = np.asarray(values)
        codes, targets = _targets(y, options)
        n_classes = int(codes.max()) + 1 if options.is_classification else 1

        keep = np.arange(len(codes), dtype="int64")
        watch = np.empty(0, dtype="int64")
        if options.early_stopping:
            watch = validation_indices(
                codes, options.validation_fraction, options.seed, stratified=options.is_classification
            )
            if watch.size:
                keep = np.setdiff1d(keep, watch)

        model = self._build(tf, X.shape[1], n_classes, options)
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=options.learning_rate),
            loss=self._loss(),
        )
        callbacks: list[Any] = []
        validation_data = None
        stopper: Any = None
        if watch.size:
            validation_data = (X[watch], targets[watch])
            stopper = tf.keras.callbacks.EarlyStopping(
                monitor="val_loss",
                patience=options.patience,
                restore_best_weights=True,
                verbose=0,
            )
            callbacks.append(stopper)
        history = model.fit(
            X[keep],
            targets[keep],
            epochs=options.epochs,
            batch_size=min(options.batch_size, max(1, len(keep))),
            validation_data=validation_data,
            callbacks=callbacks,
            shuffle=True,
            verbose=0,
        )

        losses = [float(value) for value in history.history.get("val_loss", [])]
        best_index = int(np.argmin(losses)) if losses else -1
        # `stopped_epoch` is 0 when EarlyStopping never fired, which is a
        # different thing from it firing on the last epoch.
        stopped_epoch = int(getattr(stopper, "stopped_epoch", 0) or 0)
        self._model = model
        self._n_classes = n_classes
        self.classes_ = np.unique(codes)
        self.training_report_ = _report(
            options,
            n_train=len(keep),
            n_validation=watch.size,
            n_features=int(X.shape[1]),
            epochs_run=len(history.history.get("loss", [])),
            best_epoch=best_index + 1 if losses else 0,
            best_validation_loss=min(losses) if losses else None,
            stop_reason=(
                f"the validation loss did not improve for {options.patience} epochs, "
                f"stopping at epoch {stopped_epoch}"
                if stopped_epoch
                else "reached the configured number of epochs"
            ),
            determinism={"tensorflow": determinism},
        )
        return self

    # -- predicting -------------------------------------------------------

    def predict(self, matrix: Any) -> np.ndarray:
        self._module()  # the friendly "not installed" refusal, before touching the model
        X = np.ascontiguousarray(matrix, dtype="float32")
        if self.is_classification:
            return np.asarray(self._model.predict(X, verbose=0).argmax(axis=1), dtype="int64")
        return np.asarray(self._model.predict(X, verbose=0)[:, 0], dtype="float64")

    def _predict_proba(self, matrix: Any) -> np.ndarray:
        tf = self._module()
        X = np.ascontiguousarray(matrix, dtype="float32")
        return np.asarray(
            tf.nn.softmax(self._model.predict(X, verbose=0), axis=1), dtype="float64"
        )


def tf_loss_sparse() -> Any:
    """Sparse categorical cross-entropy: the Targets are class *codes*."""
    return import_module("tensorflow").keras.losses.SparseCategoricalCrossentropy(
        from_logits=True
    )


def _targets(values: np.ndarray, options: MlpOptions) -> tuple[np.ndarray, np.ndarray]:
    """The class codes, and the framework-shaped Target to regress or learn.

    The codes are what the leaderboard scores against; the second array is what
    the network optimises (``float32`` for a regression, ``int64`` codes for a
    classification — every Model here gets its Target as class codes, so the
    cross-entropy is sparse and a class is never re-ordered).
    """
    codes = np.asarray(values)
    if not options.is_classification:
        return codes.astype("float64"), codes.astype("float32")
    if codes.dtype.kind == "f" and not float(codes.min()).is_integer():
        raise ModelError(
            "a classification Target reached the network as non-integer class codes"
        )
    codes = codes.astype("int64")
    if codes.size and codes.min() < 0:
        raise ModelError("classification class codes must be non-negative")
    return codes, codes


__all__ = [
    "DEFAULT_VALIDATION_FRACTION",
    "MlpModel",
    "MlpOptions",
    "TensorFlowMLP",
    "TorchMLP",
    "resolve_mlp_options",
    "validation_indices",
]
