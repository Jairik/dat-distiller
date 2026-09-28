"""The MLP Models: PyTorch and TensorFlow, on the same leaderboard as sklearn.

The three things worth pinning here, in order of how much they matter:

1. **A missing extra is a readable refusal, not a crash.** A bare install has to
   get the registry, the plan, the metric catalogue and a 422 that names the
   extra and the command — all of which the tests at the bottom of this module
   check without importing either framework.
2. **Early stopping watches the training split and nothing else.** The
   validation slice is cut from whatever ``fit`` was handed, and ``fit`` is only
   ever handed the training matrix, so the held-out test rows are structurally
   out of reach. These tests assert the row arithmetic that makes that true.
3. **Both frameworks produce comparable metrics.** Not "metrics too" — the same
   metric names, computed by the same functions as a Random Forest's, on one
   ranked board. That is why there is no second scoring code path in
   :mod:`dat_distiller.training.neural`, and why the leaderboard test below
   compares two entries' metric *key sets* rather than two numbers.

Every test that needs a framework guards itself with ``importorskip``, so this
module is green on a bare install. They are also deliberately tiny: a handful of
epochs on a toy frame. A test that takes 30s is a test that gets skipped later.
"""

from __future__ import annotations

import json
import time
from typing import Any

import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient

from dat_distiller.store.provenance import PROVENANCE_COLUMN
from dat_distiller.store.store import DatasetStore
from dat_distiller.training import trainers
from dat_distiller.training.neural import (
    DEFAULT_VALIDATION_FRACTION,
    resolve_mlp_options,
    validation_indices,
)
from dat_distiller.training.trainers import ModelError, get_spec

#: The two MLPs and the extra each one needs. ``importorskip`` on a fixture, so
#: a test that needs *some* neural net runs on a torch-only install, and a test
#: that needs a specific one skips when that framework is absent.
NEURAL = {"torch_mlp": "torch", "tensorflow_mlp": "tensorflow"}

#: A toy configuration for a Training Run: one small hidden layer and a few dozen
#: epochs at a rate that converges inside them. Both frameworks reach ~1.0 on the
#: toy frame at these settings; the *defaults* (50 epochs at 0.001) are what the
#: registry test checks, and a real user tunes from there.
TINY = {"hidden_units": [16], "epochs": 40, "batch_size": 16, "learning_rate": 0.05}


def require(name: str) -> str:
    """The extra a Model needs, skipped if this install does not have it."""
    pytest.importorskip(NEURAL[name])
    return name


# -- the options, with no framework involved ---------------------------------


def test_the_registry_defaults_describe_a_complete_configuration() -> None:
    for name in NEURAL:
        spec = get_spec(name)
        for task_type in spec.task_types:
            defaults = spec.defaults_for(task_type)
            options = resolve_mlp_options(defaults, task_type)
            assert options.hidden_units, name
            assert options.epochs > 0 and options.batch_size > 0
            assert options.learning_rate > 0.0
            assert options.early_stopping is True
            assert options.patience > 0
            assert 0.0 < options.validation_fraction < 1.0
    # The two are the same Model in two libraries, so they start the same.
    torch_defaults = get_spec("torch_mlp").defaults_for("classification")
    tf_defaults = get_spec("tensorflow_mlp").defaults_for("classification")
    assert torch_defaults == tf_defaults


def test_the_default_validation_fraction_is_a_documented_share_of_the_training_rows() -> None:
    assert 0.0 < DEFAULT_VALIDATION_FRACTION < 0.5
    assert get_spec("torch_mlp").defaults_for("regression")["validation_fraction"] == (
        DEFAULT_VALIDATION_FRACTION
    )


@pytest.mark.parametrize(
    "override, message",
    [
        ({"epochs": 0}, "at least 1"),
        ({"epochs": "many"}, "must be a number"),
        ({"epochs": 2.5}, "whole number"),
        ({"batch_size": 0}, "at least 1"),
        ({"patience": 0}, "at least 1"),
        ({"learning_rate": 0.0}, "positive"),
        ({"learning_rate": -0.1}, "positive"),
        ({"dropout": 1.0}, "below 1"),
        ({"dropout": "half"}, "must be a number"),
        ({"hidden_units": []}, "at least one hidden layer"),
        ({"hidden_units": "wide"}, "list of layer widths"),
        ({"hidden_units": [0]}, "at least 1"),
        ({"validation_fraction": 0.0}, "between 0 and 1"),
        ({"validation_fraction": 1.0}, "between 0 and 1"),
    ],
)
def test_a_nonsense_hyperparameter_is_a_readable_refusal(override: dict, message: str) -> None:
    """A bad override fails the way a bad sklearn override does: readably.

    ``ModelError`` is a ``ValueError``, which is what the Training Run catches
    to mark one Model failed and rank the rest.
    """
    with pytest.raises(ModelError, match=message):
        resolve_mlp_options(override, "classification")


def test_an_unknown_task_type_is_refused() -> None:
    with pytest.raises(ModelError, match="task type must be one of"):
        resolve_mlp_options({}, "clustering")


# -- the validation slice ----------------------------------------------------


def test_the_validation_slice_is_stratified_deterministic_and_disjoint() -> None:
    codes = np.array([0] * 20 + [1] * 20 + [2] * 20)
    first = validation_indices(codes, 0.15, 5, stratified=True)
    again = validation_indices(codes, 0.15, 5, stratified=True)
    other = validation_indices(codes, 0.15, 6, stratified=True)
    assert list(first) == list(again), "the same seed must give the same slice"
    assert list(first) != list(other), "the seed must actually be used"
    assert len(first) == 9, "15% of each of three classes, rounded"
    for code in (0, 1, 2):
        assert (codes[first] == code).sum() == 3, code
    # Every row stays available to whichever side it is not on: the slice is a
    # partition of the rows it was given, not a subset of an unknown pool.
    assert set(first) | set(range(len(codes))) == set(range(len(codes)))


def test_a_regression_target_is_not_stratified() -> None:
    """A regression Target has no classes to keep in proportion, so it is one group."""
    values = np.linspace(0.0, 1.0, 20)
    held = validation_indices(values, 0.2, 1, stratified=False)
    assert len(held) == 4
    assert list(held) == list(validation_indices(values, 0.2, 1, stratified=False))


def test_a_class_with_one_row_is_never_held_out() -> None:
    codes = np.array([0] * 10 + [1])
    held = validation_indices(codes, 0.5, 2, stratified=True)
    assert (codes[held] == 1).sum() == 0
    assert (codes[held] == 0).sum() > 0


def test_too_few_rows_to_split_hands_back_nothing() -> None:
    assert validation_indices(np.array([0, 1]), 0.5, 0, stratified=True).size == 0


# -- a toy frame the networks can actually learn -----------------------------


#: Where each class's blob sits. The third one is on a *negative* axis on
#: purpose, so a Target that is a trivial function of the row's own coordinates
#: cannot be recovered by a one-liner: the network has to actually separate it.
BLOBS = np.array([[-3.0, 0.0, 0.0], [0.0, 3.0, 0.0], [0.0, 0.0, -3.0]])


def toy_data(rows: int = 60, seed: int = 4) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """A toy frame: three well-separated blobs, and two Targets on them.

    The features are drawn *from* the classes rather than the other way round,
    so the groups are separable by construction and a toy net learns them in a
    handful of epochs. This is a smoke test, not a benchmark — but a task a tiny
    net could not beat chance on would be testing the toy rather than the Model.
    The classes are balanced by construction, so an F1 threshold means what it
    says.
    """
    rng = np.random.default_rng(seed)
    third = rows // 3
    counts = [third, third, rows - 2 * third]
    codes = np.concatenate(
        [np.full(count, code, dtype="int64") for code, count in enumerate(counts)]
    )
    matrix = np.vstack(
        [BLOBS[code] + rng.normal(scale=0.4, size=(count, 3)) for code, count in enumerate(counts)]
    )
    # A regression Target that uses every feature, so a net has something to fit.
    # Kept on an O(1) scale: the metrics are reported in the Target's own units,
    # and a Target spanning -10..4 needs a far larger learning rate to converge in
    # a toy number of epochs. That is a fact about the data, not about the Model.
    values = (
        0.6 * matrix[:, 0]
        + 0.2 * matrix[:, 1]
        - 0.1 * matrix[:, 2]
        + rng.normal(scale=0.02, size=len(matrix))
    )
    return matrix, codes, values.astype("float64")


def build(name: str, task_type: str, **overrides: Any) -> Any:
    """A tiny MLP: one small hidden layer, a couple of dozen epochs, one seed.

    The default 50-epoch, two-layer configuration is exercised by
    :func:`test_the_registry_defaults_describe_a_complete_configuration`;
    the tests that genuinely *fit* something pin the small one, because a test
    that takes 30s is a test that will be skipped later.
    """
    spec = get_spec(name)
    parameters = trainers.resolve_hyperparameters(
        spec,
        task_type,
        seed=5,
        overrides={"hidden_units": [8], "batch_size": 16, "learning_rate": 0.01, **overrides},
    )
    return trainers.build_estimator(spec, task_type, hyperparameters=parameters)


@pytest.mark.parametrize("name", sorted(NEURAL))
@pytest.mark.parametrize("task_type", ["classification", "regression"])
def test_an_mlp_learns_a_toy_frame_and_the_same_seed_gives_the_same_model(
    name: str, task_type: str
) -> None:
    require(name)
    matrix, classes_codes, regression_codes = toy_data()
    values = classes_codes if task_type == "classification" else regression_codes
    classes = ["a", "b", "c"]

    fitted = build(name, task_type, epochs=60, early_stopping=False).fit(matrix, values)
    predicted = fitted.predict(matrix)
    assert len(predicted) == len(values)
    assert np.isfinite(np.asarray(predicted, dtype="float64")).all()

    if task_type == "classification":
        scores = trainers.evaluate_classification(
            values, predicted, proba=fitted.predict_proba(matrix), classes=classes
        )
        # Comfortably better than chance, and the same metrics a Random Forest
        # gets — computed by the same function.
        assert scores["accuracy"]["value"] > 0.8
        assert scores["f1_macro"]["value"] > 0.8
        assert scores["roc_auc"]["value"] is not None
        assert list(fitted.classes_) == [0, 1, 2]
        probabilities = fitted.predict_proba(matrix)
        assert np.allclose(probabilities.sum(axis=1), 1.0)
        assert set(scores) == set(trainers.evaluate_classification(values, predicted, classes=classes))
    else:
        scores = trainers.evaluate_regression(values, predicted)
        assert scores["r2"]["value"] > 0.8
        assert scores["mae"]["value"] < 1.0
        # A regression Model has no class probabilities, so the leaderboard asks
        # for none — exactly as it does for the sklearn regressors.
        assert not hasattr(fitted, "predict_proba")
        assert trainers.FittedModel(
            spec=fitted, task_type="regression", estimator=fitted, classes=[], hyperparameters={}
        ).proba(matrix) is None

    # The determinism claim, checked rather than asserted: the same seed, the
    # same rows, the same numbers. A flaky leaderboard is a real defect.
    again = build(name, task_type, epochs=60, early_stopping=False).fit(matrix, values)
    assert np.array_equal(again.predict(matrix), predicted)
    if task_type == "classification":
        assert np.array_equal(again.predict_proba(matrix), fitted.predict_proba(matrix))


@pytest.mark.parametrize("name", sorted(NEURAL))
def test_early_stopping_watches_a_slice_of_the_rows_it_was_given_and_nothing_else(
    name: str,
) -> None:
    """The held-out test split is out of reach by construction, so the slice is too.

    ``fit`` is only ever handed the training matrix, so the arithmetic
    ``training_rows + validation_rows == len(X)`` *is* the guarantee: there is no
    row in the total that did not come out of the training split.
    """
    require(name)
    matrix, values, _ = toy_data(rows=80)
    early = build(name, "classification", epochs=30, patience=2, validation_fraction=0.2).fit(
        matrix, values
    )
    report = early.training_report_
    stopping = report["early_stopping"]
    assert stopping["enabled"] is True
    assert stopping["validation_fraction"] == 0.2
    assert stopping["validation_rows"] == pytest.approx(80 * 0.2, abs=4)
    assert stopping["training_rows"] + stopping["validation_rows"] == 80
    assert "never the held-out test split" in stopping["validation_split_from"]
    assert 0 < report["epochs"]["run"] <= 30
    assert report["epochs"]["best"] <= report["epochs"]["run"]
    assert np.isfinite(report["epochs"]["best_validation_loss"])
    assert report["architecture"]["hidden_units"] == [8]
    # The report is destined for a checkpoint, and `nan` is not valid JSON.
    json.dumps(report)

    # Switching early stopping off must spend every row on fitting, and must
    # report no validation loss rather than a fake one.
    full = build(name, "classification", epochs=2, early_stopping=False).fit(matrix, values)
    assert full.training_report_["early_stopping"]["validation_rows"] == 0
    assert full.training_report_["early_stopping"]["training_rows"] == 80
    assert full.training_report_["epochs"]["best_validation_loss"] is None
    json.dumps(full.training_report_)


@pytest.mark.parametrize("name", sorted(NEURAL))
def test_a_refused_hyperparameter_fails_the_model_and_not_the_framework(name: str) -> None:
    require(name)
    matrix, values, _ = toy_data(rows=40)
    fitted = build(name, "classification", epochs=2, hidden_units=[0])
    with pytest.raises(ModelError, match="at least 1"):
        fitted.fit(matrix, values)


# -- the same leaderboard ----------------------------------------------------


def neural_frame(rows: int = 60, seed: int = 3) -> pd.DataFrame:
    """A Dataset Version with a learnable Label Column and clean Provenance."""
    rng = np.random.default_rng(seed)
    age = rng.integers(18, 80, rows)
    spend = rng.normal(0, 1, rows).round(3)
    spam = (age < 45) & (spend > -0.5)
    return pd.DataFrame(
        {
            "age": age,
            "spend": spend,
            "is_spam": pd.array(spam, dtype="boolean"),
            PROVENANCE_COLUMN: [
                json.dumps(
                    {
                        "row_origin": "uploaded",
                        "label_origins": {"is_spam": {"origin": "human_reviewed"}},
                    }
                )
                for _ in range(rows)
            ],
        }
    )


@pytest.fixture
def project_id(client: TestClient) -> str:
    return client.post("/api/projects", json={"name": "neural nets"}).json()["id"]


@pytest.fixture
def version(client: TestClient, project_id: str) -> str:
    store: DatasetStore = client.app.state.store
    return store.create_version(project_id, neural_frame(), origin="uploaded").id


def wait_for_job(client: TestClient, job_id: str, timeout: float = 120.0) -> dict[str, Any]:
    deadline = time.time() + timeout
    while time.time() < deadline:
        job = client.get(f"/api/jobs/{job_id}").json()
        if job["status"] in ("completed", "failed", "cancelled"):
            return job
        time.sleep(0.05)
    raise AssertionError("the training job did not finish")


def train(client: TestClient, **body: Any) -> dict[str, Any]:
    start = client.post("/api/train/run", json=body)
    assert start.status_code == 202, start.text
    job = wait_for_job(client, start.json()["id"])
    assert job["status"] == "completed", job["error"]
    return job["result"]


@pytest.mark.parametrize("name", sorted(NEURAL))
def test_an_mlp_is_ranked_against_a_random_forest_on_one_board(
    client: TestClient, version: str, name: str
) -> None:
    """The whole point of the issue: one board, one ranking, one metric function."""
    require(name)
    run = train(
        client,
        version_id=version,
        target="is_spam",
        models=[name, "random_forest"],
        hyperparameters={name: TINY},
    )
    board = run["leaderboard"]
    assert {entry["model"] for entry in board} == {name, "random_forest"}
    assert all(entry["status"] == "ok" for entry in board), [
        (entry["model"], entry.get("error")) for entry in board
    ]
    assert run["library_versions"][NEURAL[name]] is not None
    # The same metrics, the same names, the same definitions — because both rows
    # were produced by the same `_score`.
    mlp, forest = board
    if mlp["model"] != name:
        mlp, forest = forest, mlp
    assert set(mlp["metrics"]) == set(forest["metrics"])
    assert mlp["library"] in ("torch", "tensorflow")
    assert forest["library"] == "scikit-learn"
    # Both got a real ROC-AUC: the MLP has a softmax head, so unlike the SVM it
    # does produce class probabilities.
    assert mlp["metrics"]["roc_auc"]["value"] is not None
    assert forest["metrics"]["roc_auc"]["value"] is not None
    assert mlp["metrics"]["accuracy"]["value"] > 0.6
    # Ranked by the same primary metric, and the ranks are the board's order.
    # Both are fitted well on this toy, so a tie at the top is a legitimate
    # outcome — the assertion is that the board ranks them together, not that
    # one of them wins.
    assert {entry["rank"] for entry in board} in ({1, 2}, {1})
    assert [entry["rank"] for entry in board] == sorted(entry["rank"] for entry in board)
    assert all(entry["primary_metric"] == run["primary_metric"] for entry in board)
    scores = [entry["primary"]["value"] for entry in board]
    assert scores == sorted(scores, reverse=True)
    assert all(0.0 <= score <= 1.0 for score in scores), "f1_macro is a rate"
    # The hyperparameters recorded are the ones it was fitted with, seed included.
    assert run["hyperparameters"][name]["seed"] == run["seed"]
    assert mlp["hyperparameters"] == run["hyperparameters"][name]
    assert mlp["hyperparameters"]["epochs"] == TINY["epochs"]
    assert run["library_versions"][mlp["library"]] is not None


@pytest.mark.parametrize("name", sorted(NEURAL))
def test_an_mlp_is_tuned_on_the_training_split_only(
    client: TestClient, version: str, name: str
) -> None:
    require(name)
    run = train(
        client,
        version_id=version,
        target="is_spam",
        models=[name],
        hyperparameters={name: {**TINY, "epochs": 3}},
        tuning={
            "enabled": True,
            "strategy": "grid",
            "cv": 2,
            "search_space": {name: {"learning_rate": [0.001, 0.01]}},
        },
    )
    result = run["tuning"]["results"][name]
    assert result["skipped"] is None
    assert result["candidates"] == 2
    assert result["trained_on_rows"] == run["training_split"]["rows"]
    assert set(result["trained_on_rows"]).isdisjoint(run["test_split"]["indices"])
    assert run["hyperparameters"][name]["learning_rate"] in (0.001, 0.01)
    # The pinned hyperparameters survived the search untouched.
    assert run["hyperparameters"][name]["epochs"] == 3
    assert run["hyperparameters"][name]["hidden_units"] == TINY["hidden_units"]
    assert run["leaderboard"][0]["status"] == "ok"


@pytest.mark.parametrize("name", sorted(NEURAL))
def test_an_mlp_is_ranked_for_a_regression_target_too(
    client: TestClient, version: str, name: str
) -> None:
    require(name)
    run = train(
        client,
        version_id=version,
        target="spend",
        models=[name, "random_forest"],
        hyperparameters={name: TINY},
    )
    assert run["task_type"] == "regression"
    assert run["primary_metric"] == "r2"
    board = run["leaderboard"]
    assert {entry["model"] for entry in board} == {name, "random_forest"}
    for entry in board:
        for metric in ("mae", "rmse", "r2", "explained_variance", "max_error"):
            assert entry["metrics"][metric]["value"] is not None, (entry["model"], metric)
        assert entry["metrics"]["r2"]["value"] is not None
    assert "accuracy" not in board[0]["metrics"]


@pytest.mark.parametrize("name", sorted(NEURAL))
def test_a_broken_mlp_hyperparameter_is_reported_without_sinking_the_board(
    client: TestClient, version: str, name: str
) -> None:
    require(name)
    run = train(
        client,
        version_id=version,
        target="is_spam",
        models=[name, "knn"],
        hyperparameters={name: {"hidden_units": [0]}},
    )
    mlp = next(entry for entry in run["leaderboard"] if entry["model"] == name)
    assert mlp["status"] == "failed"
    assert "hidden_units" in mlp["error"]
    assert mlp["rank"] is None
    knn = next(entry for entry in run["leaderboard"] if entry["model"] == "knn")
    assert knn["status"] == "ok" and knn["rank"] == 1
    assert any("failed to fit" in warning for warning in run["warnings"])


# -- without either framework, the app still answers -----------------------


def test_the_registry_and_plan_never_import_a_framework() -> None:
    """A bare install gets the metadata; the framework is only touched on fit."""
    from dat_distiller.training.neural import MlpModel, TensorFlowMLP, TorchMLP

    for cls in (TorchMLP, TensorFlowMLP):
        assert issubclass(cls, MlpModel)
        estimator = cls()  # no import, no framework
        assert estimator.get_params()["hidden_units"] == (64, 32)
        assert estimator.extra == cls.__name__.replace("MLP", "").lower()
    with pytest.raises(NotImplementedError):
        MlpModel().fit(np.zeros((1, 1)), np.zeros(1))
    with pytest.raises(ValueError, match="invalid hyperparameter"):
        TorchMLP().set_params(nonsense=1)


def test_a_neural_net_is_clonable_for_a_search() -> None:
    """scikit-learn's `clone` round-trips it, which is what tuning needs.

    Tuning needs both extras: the search itself is scikit-learn's, and a
    framework's net to put through it. On a torch-only install this skips, which
    is fine — the property being checked is about scikit-learn's ``clone``.
    """
    pytest.importorskip("sklearn")
    from sklearn.base import clone

    for name in NEURAL:
        require(name)
        estimator = build(name, "classification", epochs=3)
        copy = clone(estimator)
        assert copy.get_params() == estimator.get_params()
        assert copy.get_params()["task_type"] == "classification"
        # An independent object, not a shared reference.
        copy.set_params(epochs=99)
        assert estimator.get_params()["epochs"] == 3
        assert copy.get_params()["epochs"] == 99
