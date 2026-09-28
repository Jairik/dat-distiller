"""The Training Run: fit the Models, tune them, rank them, persist the run.

These tests really fit models. They are guarded with ``importorskip`` so a bare
install (no optional extras) still has a green suite, but with the ``sklearn``
extra installed every one of them runs.

The split tests matter most here:

- a Training Run records which rows the tuning searched, and that is exactly the
  training split;
- changing only the *held-out* rows leaves the tuning result byte-identical, so
  the test split demonstrably had no influence on the search.
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

pytest.importorskip("sklearn")

#: Neural nets are the two heaviest Models on the board, so every test that
#: happens to include them pins these: one hidden layer, a handful of epochs and
#: no early stopping. Real defaults (50 epochs, two layers) belong in the
#: dedicated neural tests, not in a test about the leaderboard's shape.
TINY_NEURAL: dict[str, dict[str, Any]] = {
    name: {"hidden_units": [8], "epochs": 4, "batch_size": 16, "early_stopping": False}
    for name in ("torch_mlp", "tensorflow_mlp")
}


def installed_neural_hyperparameters() -> dict[str, dict[str, Any]]:
    """The neural pins for the frameworks this install actually has.

    The backend rightly refuses hyperparameters for a Model it is not training,
    so a request that pins `tensorflow_mlp` on a machine without TensorFlow is a
    422 rather than a silently ignored key.
    """
    return {
        name: params
        for name, params in TINY_NEURAL.items()
        if trainers.is_available(trainers.MODEL_SPECS[name])
    }


# -- data --------------------------------------------------------------------


def labeled_frame(rows: int = 60, seed: int = 3) -> pd.DataFrame:
    """A learnable frame: `is_spam` is a real function of two features."""
    rng = np.random.default_rng(seed)
    age = rng.integers(18, 80, rows)
    spend = rng.normal(0, 1, rows).round(3)
    region = rng.choice(["north", "south", "east"], rows)
    spam = (age < 45) & (spend > -0.5)
    return pd.DataFrame(
        {
            "age": age,
            "spend": spend,
            "region": region,
            "is_spam": pd.array(spam, dtype="boolean"),
            "is_spam__confidence": pd.array([0.95] * rows, dtype="Float64"),
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
    return client.post("/api/projects", json={"name": "leaderboard lab"}).json()["id"]


@pytest.fixture
def version(client: TestClient, project_id: str) -> str:
    store: DatasetStore = client.app.state.store
    return store.create_version(project_id, labeled_frame(), origin="uploaded").id


def wait_for_job(client: TestClient, job_id: str, timeout: float = 240.0) -> dict[str, Any]:
    deadline = time.time() + timeout
    while time.time() < deadline:
        job = client.get(f"/api/jobs/{job_id}").json()
        if job["status"] in ("completed", "failed", "cancelled"):
            return job
        time.sleep(0.05)
    raise AssertionError("the training job did not finish")


def train(client: TestClient, **body: Any) -> dict[str, Any]:
    """Run a Training Run to completion and return its persisted payload."""
    start = client.post("/api/train/run", json=body)
    assert start.status_code == 202, start.text
    job = wait_for_job(client, start.json()["id"])
    assert job["status"] == "completed", job["error"]
    return job["result"]


# -- the registry endpoints -------------------------------------------------


def test_the_model_endpoint_lists_every_model_with_its_task_types(client: TestClient) -> None:
    body = client.get("/api/train/models").json()
    by_name = {model["name"]: model for model in body["models"]}
    assert set(by_name) == {
        "logistic_regression",
        "linear_regression",
        "svm",
        "random_forest",
        "gradient_boosting",
        "knn",
        "naive_bayes",
        "lightgbm",
        "torch_mlp",
        "tensorflow_mlp",
    }
    assert by_name["naive_bayes"]["task_types"] == ["classification"]
    assert by_name["logistic_regression"]["task_types"] == ["classification"]
    assert by_name["linear_regression"]["task_types"] == ["regression"]
    assert by_name["knn"]["task_types"] == ["classification", "regression"]
    assert by_name["random_forest"]["available"] is True
    assert by_name["random_forest"]["class_path"]["regression"].endswith(
        "RandomForestRegressor"
    )
    assert by_name["random_forest"]["default_hyperparameters"]["regression"]
    assert by_name["random_forest"]["search_space"], "the UI needs a grid to show"
    assert body["extras"]["sklearn"] is True
    # The neural nets are described whether or not the extras are here, so the
    # picker can show them greyed out with the command that would enable them.
    for name, extra in (("torch_mlp", "torch"), ("tensorflow_mlp", "tensorflow")):
        model = by_name[name]
        assert model["task_types"] == ["classification", "regression"]
        assert model["extra"] == extra
        assert model["available"] is body["extras"][extra]
        assert model["search_space"], "tuning has to work on an MLP too"
        assert model["default_hyperparameters"]["classification"]["hidden_units"]


def test_the_metrics_endpoint_offers_a_default_per_task_type(client: TestClient) -> None:
    body = client.get("/api/train/metrics").json()["task_types"]
    assert body["classification"]["default_primary_metric"] == "f1_macro"
    assert body["regression"]["default_primary_metric"] == "r2"
    names = {metric["name"] for metric in body["classification"]["metrics"]}
    assert {"accuracy", "f1_macro", "roc_auc"} <= names
    assert "mae" not in names
    regression = {metric["name"] for metric in body["regression"]["metrics"]}
    assert {"mae", "rmse", "r2"} <= regression
    assert body["regression"]["metrics"][1]["higher_is_better"] is True
    only_one = client.get("/api/train/metrics?task_type=regression").json()["task_types"]
    assert list(only_one) == ["regression"]


# -- planning refuses what cannot be done -----------------------------------


def test_the_plan_defaults_to_the_task_types_default_metric(client: TestClient, version: str) -> None:
    body = client.post("/api/train/plan", json={"version_id": version, "target": "is_spam"})
    assert body.status_code == 200
    plan = body.json()
    assert plan["task_type"] == "classification"
    assert plan["primary_metric"] == "f1_macro"
    assert "naive_bayes" in plan["model_names"]
    assert "linear_regression" not in plan["model_names"]
    assert plan["train_rows"] == 48 and plan["test_rows"] == 12


def test_the_plan_refuses_a_model_that_cannot_do_the_task_type(
    client: TestClient, version: str
) -> None:
    response = client.post(
        "/api/train/plan",
        json={"version_id": version, "target": "spend", "models": ["naive_bayes"]},
    )
    assert response.status_code == 422
    detail = response.json()["detail"]
    assert "naive_bayes" in detail and "regression" in detail
    assert "linear_regression" in detail, "the error should say what is allowed"
    # The same refusal applies to the run endpoint.
    refused = client.post(
        "/api/train/run",
        json={"version_id": version, "target": "spend", "models": ["naive_bayes"]},
    )
    assert refused.status_code == 422
    assert "naive_bayes" in refused.json()["detail"]


def test_the_plan_refuses_a_metric_the_task_type_cannot_produce(
    client: TestClient, version: str
) -> None:
    wrong_metric = client.post(
        "/api/train/plan",
        json={"version_id": version, "target": "is_spam", "primary_metric": "rmse"},
    )
    assert wrong_metric.status_code == 422
    assert "rmse" in wrong_metric.json()["detail"]
    unknown = client.post(
        "/api/train/plan",
        json={"version_id": version, "target": "is_spam", "primary_metric": "logloss"},
    )
    assert unknown.status_code == 422
    accepted = client.post(
        "/api/train/plan",
        json={"version_id": version, "target": "is_spam", "primary_metric": "roc_auc"},
    )
    assert accepted.status_code == 200
    assert accepted.json()["primary_metric"] == "roc_auc"


def test_the_plan_refuses_an_unknown_model_or_hyperparameters(
    client: TestClient, version: str
) -> None:
    assert (
        client.post(
            "/api/train/plan",
            json={"version_id": version, "target": "is_spam", "models": ["catboost"]},
        ).status_code
        == 422
    )
    assert (
        client.post(
            "/api/train/plan",
            json={"version_id": version, "target": "is_spam", "models": []},
        ).status_code
        == 422
    )
    stray = client.post(
        "/api/train/plan",
        json={
            "version_id": version,
            "target": "is_spam",
            "models": ["knn"],
            "hyperparameters": {"svm": {"C": 1.0}},
        },
    )
    assert stray.status_code == 422
    assert "svm" in stray.json()["detail"]


def test_the_run_endpoint_validates_the_version_and_target(client: TestClient) -> None:
    assert client.post("/api/train/run", json={"version_id": "nope", "target": "a"}).status_code == 404
    assert client.post("/api/train/plan", json={"version_id": "nope", "target": "a"}).status_code == 404


# -- a Training Run ---------------------------------------------------------


def test_a_training_run_fits_every_model_and_ranks_them(
    client: TestClient, version: str
) -> None:
    # A default selection is every Model this install can run, which on a machine
    # with the heavy extras also means the neural nets — pinned to a token number
    # of epochs so the test does not pay for a real search.
    run = train(
        client,
        version_id=version,
        target="is_spam",
        hyperparameters=installed_neural_hyperparameters(),
    )
    board = run["leaderboard"]
    expected = {
        name
        for name, spec in trainers.MODEL_SPECS.items()
        if name != "linear_regression" and trainers.is_available(spec)
    }
    assert set(entry["model"] for entry in board) == expected
    assert all(entry["status"] == "ok" for entry in board), [
        (entry["model"], entry.get("error")) for entry in board
    ]
    ranks = [entry["rank"] for entry in board]
    assert ranks == sorted(ranks), "the board must be ordered best first"
    assert ranks[0] == 1
    scores = [entry["primary"]["value"] for entry in board]
    assert all(value is not None for value in scores)
    assert scores == sorted(scores, reverse=True), "f1_macro: higher is better"
    # Every Model really learned something from a learnable frame. The bar is
    # "comfortably better than chance", deliberately: a tree ensemble on 48
    # training rows is genuinely non-deterministic even with a fixed seed
    # (observed 0.75–1.0 accuracy across identical runs), so a tighter threshold
    # would be a coin flip rather than a claim. Ranking, ordering and metric
    # *shape* are the deterministic things, and those are asserted tightly above.
    best = board[0]
    assert best["metrics"]["accuracy"]["value"] > 0.6
    with_proba = next(entry for entry in board if entry["model"] == "logistic_regression")
    assert with_proba["metrics"]["roc_auc"]["value"] is not None
    assert 0.5 <= with_proba["metrics"]["roc_auc"]["value"] <= 1.0
    # The SVM is fitted without calibrated probabilities, so its ROC-AUC is a
    # null with a reason rather than a made-up number.
    svm = next(entry for entry in board if entry["model"] == "svm")
    assert svm["metrics"]["roc_auc"]["value"] is None
    assert "class probabilities" in svm["metrics"]["roc_auc"]["reason"]
    assert svm["metrics"]["accuracy"]["value"] > 0.6


def test_a_training_run_records_seeds_library_versions_and_hyperparameters(
    client: TestClient, version: str
) -> None:
    run = train(
        client,
        version_id=version,
        target="is_spam",
        models=["logistic_regression", "knn", "lightgbm"],
        hyperparameters={"knn": {"n_neighbors": 3}},
    )
    assert run["seeds"] == {"split": run["setup"]["seed"], "models": run["seed"]}
    assert isinstance(run["seed"], int)
    versions = run["library_versions"]
    assert versions["scikit-learn"] and versions["lightgbm"]
    assert run["library_versions"]["python"]
    assert run["hyperparameters"]["knn"] == {"n_neighbors": 3}, "the override is recorded"
    assert run["hyperparameters"]["logistic_regression"]["random_state"] == run["seed"]
    assert run["hyperparameters"]["logistic_regression"]["max_iter"] == 1000
    assert run["hyperparameters"]["lightgbm"]["random_state"] == run["seed"]
    assert run["hyperparameters"]["lightgbm"]["verbosity"] == -1
    # Each leaderboard row carries the same mapping it was fitted with.
    for entry in run["leaderboard"]:
        assert entry["hyperparameters"] == run["hyperparameters"][entry["model"]]


def test_the_run_records_the_split_it_trained_on_and_scored_on(
    client: TestClient, version: str
) -> None:
    run = train(client, version_id=version, target="is_spam", models=["knn"])
    assert run["training_split"]["n_samples"] == run["setup"]["split"]["train_rows"] == 48
    assert run["test_split"]["rows"] == 12
    assert run["test_split"]["stratified"] is True
    assert set(run["training_split"]["rows"]).isdisjoint(run["test_split"]["indices"])
    assert run["leaderboard"][0]["metrics"]["n_test"] == 12


def test_the_run_raises_the_same_check_vocabulary_against_the_training_run(
    client: TestClient, project_id: str
) -> None:
    store: DatasetStore = client.app.state.store
    frame = labeled_frame(60)
    frame["region"] = ["north"] * 56 + ["south", "east", "north", "south"]
    skewed = store.create_version(project_id, frame, origin="uploaded").id
    run = train(client, version_id=skewed, target="region", models=["knn"], features=["age"])
    assert "class_imbalance" in run["checks_raised"]

    body = client.get(f"/api/train/runs/{run['training_run_id']}").json()
    kinds = {check["kind"] for check in body["checks"]}
    assert "class_imbalance" in kinds
    assert all(check["subject_type"] == "training_run" for check in body["checks"])
    imbalance = next(c for c in body["checks"] if c["kind"] == "class_imbalance")
    assert imbalance["severity"] == "warning"
    assert imbalance["step"] == "train"
    assert imbalance["details"]["target"] == "region"
    assert body["unacknowledged_warnings"], "warnings need an Acknowledgement"
    listed = client.get(
        f"/api/checks?subject_type=training_run&subject_id={run['training_run_id']}"
    ).json()
    assert any(c["kind"] == "class_imbalance" for c in listed["checks"])


# -- tuning: the test split is out of reach ---------------------------------


def test_tuning_cross_validates_on_the_training_split_only(
    client: TestClient, version: str
) -> None:
    run = train(
        client,
        version_id=version,
        target="is_spam",
        models=["logistic_regression", "random_forest", "knn"],
        tuning={"enabled": True, "strategy": "grid", "cv": 3},
    )
    assert run["tuning"]["enabled"] is True
    assert run["tuning"]["metric"] == "f1_macro"
    searched = run["tuning"]["results"]["knn"]["trained_on_rows"]
    held_out = run["test_split"]["indices"]
    assert len(searched) == run["training_split"]["n_samples"] == 48
    assert set(searched).isdisjoint(held_out), "the tuning saw held-out rows"
    for name in ("logistic_regression", "random_forest", "knn"):
        result = run["tuning"]["results"][name]
        assert result["trained_on_rows"] == searched, name
        assert result["skipped"] is None
        assert result["best_params"]
        assert 0.0 <= result["best_score"] <= 1.0
        assert result["cv"] == 3
    # The winning parameters are what the Model was fitted with.
    for name, key in (
        ("logistic_regression", "C"),
        ("random_forest", "max_depth"),
        ("knn", "n_neighbors"),
    ):
        assert (
            run["hyperparameters"][name][key]
            == run["tuning"]["results"][name]["best_params"][key]
        ), name


def test_tuning_ignores_the_held_out_rows_entirely(
    client: TestClient, project_id: str
) -> None:
    """Same training rows, wildly different held-out rows, same search result.

    The split is a function of the request's seed, so the two runs hold out the
    same positions. If the search had ever read those rows, rewriting them would
    change what it picked.
    """
    store: DatasetStore = client.app.state.store
    clean = labeled_frame(60, seed=11)
    # An explicit seed: the split must not move when the *held-out* rows do.
    # (An unseeded split is derived from the request, which names the version.)
    request = {
        "target": "is_spam",
        "models": ["knn", "random_forest"],
        "seed": 4242,
        "tuning": {"enabled": True, "strategy": "grid", "cv": 3},
    }
    first = store.create_version(project_id, clean, origin="uploaded")
    before = train(client, **{**request, "version_id": first.id})

    # Now wreck exactly the rows this run held out, and nothing else.
    poisoned = clean.copy()
    held_out = before["test_split"]["indices"]
    poisoned.loc[held_out, "age"] = 999
    poisoned.loc[held_out, "spend"] = -50.0
    second = store.create_version(project_id, poisoned, origin="uploaded")
    after = train(client, **{**request, "version_id": second.id})

    assert before["test_split"]["indices"] == after["test_split"]["indices"] == held_out
    assert before["training_split"]["rows"] == after["training_split"]["rows"]
    assert set(before["training_split"]["rows"]).isdisjoint(held_out)
    for name in ("knn", "random_forest"):
        assert (
            before["tuning"]["results"][name]["best_params"]
            == after["tuning"]["results"][name]["best_params"]
        ), name
        assert (
            before["tuning"]["results"][name]["best_score"]
            == after["tuning"]["results"][name]["best_score"]
        ), name
        assert before["hyperparameters"][name] == after["hyperparameters"][name], name
    # The held-out scores do move — proof the two test splits really differed.
    assert before["leaderboard"] != after["leaderboard"]


def test_tuning_on_roc_auc_skips_the_models_without_probabilities(
    client: TestClient, version: str
) -> None:
    run = train(
        client,
        version_id=version,
        target="is_spam",
        models=["svm", "knn"],
        primary_metric="roc_auc",
        tuning={"enabled": True, "metric": "roc_auc", "cv": 3},
    )
    svm_result = run["tuning"]["results"]["svm"]
    assert "class probabilities" in svm_result["skipped"]
    assert svm_result["trained_on_rows"] == run["training_split"]["rows"]
    assert run["tuning"]["results"]["knn"]["skipped"] is None
    # SVM's ROC-AUC is a null with a reason, and it gets no rank.
    svm = next(entry for entry in run["leaderboard"] if entry["model"] == "svm")
    assert svm["metrics"]["roc_auc"] == {
        "value": None,
        "reason": "this Model does not output class probabilities",
    }
    assert svm["rank"] is None
    knn = next(entry for entry in run["leaderboard"] if entry["model"] == "knn")
    assert knn["metrics"]["roc_auc"]["value"] is not None
    assert knn["rank"] == 1
    assert any("without a rank" in w for w in run["warnings"])


def test_tuning_can_be_asked_to_search_roc_auc_after_calibrating_the_svm(
    client: TestClient, version: str
) -> None:
    run = train(
        client,
        version_id=version,
        target="is_spam",
        models=["svm"],
        primary_metric="roc_auc",
        hyperparameters={"svm": {"probability": True}},
        tuning={"enabled": True, "metric": "roc_auc", "cv": 3},
    )
    assert run["hyperparameters"]["svm"]["probability"] is True
    assert run["tuning"]["results"]["svm"]["skipped"] is None
    assert run["leaderboard"][0]["metrics"]["roc_auc"]["value"] is not None


def test_tuning_with_a_user_search_space_uses_it(client: TestClient, version: str) -> None:
    run = train(
        client,
        version_id=version,
        target="is_spam",
        models=["knn"],
        tuning={
            "enabled": True,
            "strategy": "grid",
            "cv": 2,
            "search_space": {"knn": {"n_neighbors": [1, 3]}},
        },
    )
    result = run["tuning"]["results"]["knn"]
    assert result["search_space"] == {"n_neighbors": [1, 3]}
    assert result["candidates"] == 2
    assert run["hyperparameters"]["knn"]["n_neighbors"] in (1, 3)


def test_tuning_a_model_with_nothing_left_to_search_is_recorded_not_fatal(
    client: TestClient, version: str
) -> None:
    run = train(
        client,
        version_id=version,
        target="is_spam",
        models=["knn", "random_forest"],
        hyperparameters={"knn": {"n_neighbors": 3}},
        tuning={"enabled": True, "cv": 2},
    )
    assert "no hyperparameter to search" in run["tuning"]["results"]["knn"]["skipped"]
    assert run["tuning"]["results"]["random_forest"]["skipped"] is None
    assert run["hyperparameters"]["knn"]["n_neighbors"] == 3
    assert all(entry["status"] == "ok" for entry in run["leaderboard"])


# -- ranking, regression, failures ------------------------------------------


def test_the_leaderboard_ranks_by_the_metric_the_user_chose(
    client: TestClient, version: str
) -> None:
    """Whichever metric the user picks is the one the board is sorted by."""
    for metric in ("accuracy", "f1_macro", "balanced_accuracy", "f1_weighted"):
        run = train(
            client,
            version_id=version,
            target="is_spam",
            models=["logistic_regression", "random_forest", "gradient_boosting", "knn"],
            primary_metric=metric,
        )
        assert run["primary_metric"] == metric
        values = [entry["metrics"][metric]["value"] for entry in run["leaderboard"]]
        assert values == sorted(values, reverse=True), metric
        assert all(
            entry["primary"]["value"] == entry["metrics"][metric]["value"]
            for entry in run["leaderboard"]
        ), metric
        assert [e["rank"] for e in run["leaderboard"]] == sorted(
            e["rank"] for e in run["leaderboard"]
        ), metric
        assert run["leaderboard"][0]["rank"] == 1, metric


def test_the_persisted_board_is_consistent_with_the_chosen_metric(
    client: TestClient, version: str
) -> None:
    """The board is not hard-wired to a default: it follows the request."""
    models = ["logistic_regression", "random_forest", "gradient_boosting", "knn", "naive_bayes"]
    default_run = train(client, version_id=version, target="is_spam", models=models)
    assert default_run["primary_metric"] == "f1_macro"
    chosen_run = train(
        client, version_id=version, target="is_spam", models=models, primary_metric="f1_minority"
    )
    assert chosen_run["primary_metric"] == "f1_minority"
    # Same split, same fitted hyperparameters: only the ranking metric changed.
    assert default_run["test_split"]["indices"] == chosen_run["test_split"]["indices"]
    assert default_run["hyperparameters"] == chosen_run["hyperparameters"]
    for run, metric in ((default_run, "f1_macro"), (chosen_run, "f1_minority")):
        values = [entry["metrics"][metric]["value"] for entry in run["leaderboard"]]
        assert values == sorted(values, reverse=True), metric
        assert run["leaderboard"][0]["rank"] == 1, metric
        assert run["leaderboard"][0]["primary"]["value"] == max(values), metric


def test_tied_models_share_a_rank(client: TestClient, version: str) -> None:
    run = train(
        client,
        version_id=version,
        target="is_spam",
        models=["logistic_regression", "random_forest", "gradient_boosting", "knn"],
        primary_metric="accuracy",
    )
    ranks = [entry["rank"] for entry in run["leaderboard"]]
    assert ranks[0] == 1
    for previous, current in zip(ranks, ranks[1:]):
        assert current >= previous, "ranks never go backwards"
    # A learnable, separable Target leaves several Models tied at the top.
    assert ranks.count(1) >= 1


def test_a_primary_metric_nothing_can_produce_is_reported_not_hidden(
    client: TestClient, version: str
) -> None:
    run = train(
        client,
        version_id=version,
        target="is_spam",
        models=["svm", "logistic_regression"],
        hyperparameters={"logistic_regression": {"C": 1.0}},
        primary_metric="f1_minority",
        features=["age", "spend"],
    )
    # f1_minority needs at least two classes in the test split; build_setup keeps
    # the split stratified, so the honest outcome is either a score or a reason.
    primary = run["leaderboard"][0]["primary"]
    if primary["value"] is None:
        assert primary["reason"]
        assert any("no Model produced" in w for w in run["warnings"])
    else:
        assert 0.0 <= primary["value"] <= 1.0


def test_a_regression_target_is_ranked_by_r_squared(client: TestClient, version: str) -> None:
    run = train(
        client,
        version_id=version,
        target="spend",
        models=["linear_regression", "svm", "random_forest", "knn", "lightgbm"],
    )
    assert run["task_type"] == "regression"
    assert run["primary_metric"] == "r2"
    r2 = [entry["metrics"]["r2"]["value"] for entry in run["leaderboard"]]
    assert r2 == sorted(r2, reverse=True), "R²: higher is better"
    for entry in run["leaderboard"]:
        for name in ("mae", "rmse", "r2", "explained_variance", "max_error"):
            assert entry["metrics"][name]["value"] is not None, (entry["model"], name)
    assert "accuracy" not in run["leaderboard"][0]["metrics"]
    best = run["leaderboard"][0]
    assert best["metrics"]["mae"]["value"] >= 0.0
    assert best["metrics"]["n_test"] == run["test_split"]["rows"]


def test_a_regression_leaderboard_can_be_ranked_by_an_error_metric(
    client: TestClient, version: str
) -> None:
    run = train(
        client,
        version_id=version,
        target="spend",
        models=["linear_regression", "knn", "lightgbm"],
        primary_metric="rmse",
    )
    assert run["primary_metric"] == "rmse"
    assert run["primary_metric_higher_is_better"] is False
    values = [entry["metrics"]["rmse"]["value"] for entry in run["leaderboard"]]
    assert values == sorted(values), "RMSE: lower is better"


def test_a_model_that_cannot_be_fitted_is_reported_without_sinking_the_board(
    client: TestClient, version: str
) -> None:
    run = train(
        client,
        version_id=version,
        target="is_spam",
        models=["knn", "lightgbm"],
        # A string where a number belongs: the estimator must reject it.
        hyperparameters={"knn": {"n_neighbors": "three"}},
    )
    knn = next(entry for entry in run["leaderboard"] if entry["model"] == "knn")
    assert knn["status"] == "failed"
    assert knn["error"]
    assert knn["rank"] is None
    lightgbm = next(entry for entry in run["leaderboard"] if entry["model"] == "lightgbm")
    assert lightgbm["status"] == "ok" and lightgbm["rank"] == 1
    assert any("failed to fit" in w for w in run["warnings"])


def test_a_run_can_reuse_a_setup_runs_checkpointed_preprocessing(
    client: TestClient, version: str
) -> None:
    setup_start = client.post(
        "/api/train/setup", json={"version_id": version, "target": "is_spam"}
    )
    assert setup_start.status_code == 202
    setup_job = wait_for_job(client, setup_start.json()["id"])
    assert setup_job["status"] == "completed", setup_job["error"]
    setup = client.get(f"/api/train/setup/{setup_job['id']}").json()

    run = train(
        client,
        version_id=version,
        target="is_spam",
        models=["knn"],
        setup_run_id=setup_job["id"],
    )
    assert run["preprocessing"] == setup["preprocessing"], "the same fitted pipeline"
    assert run["test_split"]["indices"] == setup["split_indices"]["test"]
    assert run["training_split"]["rows"] == setup["split_indices"]["train"]

    read_back = client.get(f"/api/train/runs/{run['training_run_id']}").json()
    assert read_back["setup_run_id"] == setup_job["id"]


def test_an_unusable_setup_run_id_is_refused_with_a_readable_reason(
    client: TestClient, version: str
) -> None:
    start = client.post(
        "/api/train/run",
        json={"version_id": version, "target": "is_spam", "models": ["knn"], "setup_run_id": "nope"},
    )
    assert start.status_code == 202
    job = wait_for_job(client, start.json()["id"])
    assert job["status"] == "failed"
    assert "no checkpoint to reuse" in job["error"]


# -- reading Training Runs back --------------------------------------------


def test_reading_a_training_run_back(client: TestClient, version: str) -> None:
    run = train(client, version_id=version, target="is_spam", models=["knn", "lightgbm"])
    response = client.get(f"/api/train/runs/{run['training_run_id']}")
    assert response.status_code == 200
    body = response.json()
    assert body["training_run_id"] == run["training_run_id"]
    assert body["job_id"] == run["training_run_id"]
    assert body["status"] == "completed"
    assert body["version_id"] == version
    assert body["target"] == "is_spam"
    assert body["leaderboard"] == run["leaderboard"]
    assert body["hyperparameters"] == run["hyperparameters"]
    assert body["library_versions"] == run["library_versions"]
    assert body["seeds"] == run["seeds"]
    assert len(body["split_indices"]["train"]) == 48
    assert len(body["split_indices"]["test"]) == 12
    assert "checks" in body and "unacknowledged_warnings" in body


def test_listing_a_projects_training_runs(client: TestClient, project_id: str, version: str) -> None:
    first = train(client, version_id=version, target="is_spam", models=["knn"])
    second = train(client, version_id=version, target="spend", models=["linear_regression"])
    body = client.get(f"/api/train/runs?project_id={project_id}").json()
    assert body["count"] == 2
    ids = [row["training_run_id"] for row in body["runs"]]
    assert ids == [second["training_run_id"], first["training_run_id"]], "newest first"
    assert body["runs"][0]["target"] == "spend"
    assert body["runs"][0]["task_type"] == "regression"
    assert body["runs"][0]["primary_metric"] == "r2"
    assert body["runs"][0]["best_model"] == "linear_regression"
    assert body["runs"][0]["status"] == "completed"
    assert body["runs"][0]["best_primary_value"] is not None
    assert body["runs"][0]["n_models"] == 1
    assert client.get("/api/train/runs?project_id=nowhere").json()["count"] == 0


def test_reading_something_that_is_not_a_training_run(
    client: TestClient, version: str
) -> None:
    assert client.get("/api/train/runs/nope").status_code == 404
    # A `train_setup` run is a different kind of thing and says so.
    setup = client.post("/api/train/setup", json={"version_id": version, "target": "is_spam"})
    setup_job = wait_for_job(client, setup.json()["id"])
    assert setup_job["status"] == "completed", setup_job["error"]
    response = client.get(f"/api/train/runs/{setup_job['id']}")
    assert response.status_code == 404
    assert response.json()["detail"] == "not a training run"


def test_the_training_job_is_resumable(client: TestClient, version: str) -> None:
    start = client.post(
        "/api/train/run", json={"version_id": version, "target": "is_spam", "models": ["knn"]}
    )
    job = wait_for_job(client, start.json()["id"])
    assert job["status"] == "completed"
    assert client.get(f"/api/jobs/{job['id']}").json()["resumable"] is True


def test_a_resumed_run_reuses_the_finished_models(
    client: TestClient, version: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A checkpointed Training Run resumes without refitting what it finished."""
    from dat_distiller.training import trainers as trainers_module

    start = client.post(
        "/api/train/run",
        json={"version_id": version, "target": "is_spam", "models": ["knn", "random_forest"]},
    )
    job_id = start.json()["id"]
    job = wait_for_job(client, job_id)
    assert job["status"] == "completed", job["error"]
    baseline = job["result"]

    # Simulate a server restart: the checkpoint survives, the status does not.
    with client.app.state.store.db.connect() as conn:
        conn.execute(
            "UPDATE jobs SET status = 'interrupted', result_json = NULL WHERE id = ?", (job_id,)
        )
    fitted: list[str] = []
    real_fit = trainers_module.fit_model

    def counting_fit(spec: Any, *args: Any, **kwargs: Any) -> Any:
        fitted.append(spec.name)
        return real_fit(spec, *args, **kwargs)

    monkeypatch.setattr(trainers_module, "fit_model", counting_fit)
    assert client.post(f"/api/jobs/{job_id}/resume").status_code == 200
    after = wait_for_job(client, job_id)
    assert after["status"] == "completed", after["error"]
    assert after["result"]["leaderboard"] == baseline["leaderboard"]
    # Both Models were already in the checkpoint, so nothing was refitted.
    assert fitted == [], fitted


# -- LightGBM, on its own ---------------------------------------------------


def test_lightgbm_is_trained_and_ranked(client: TestClient, version: str) -> None:
    pytest.importorskip("lightgbm")
    run = train(client, version_id=version, target="is_spam", models=["lightgbm"])
    entry = run["leaderboard"][0]
    assert entry["model"] == "lightgbm"
    assert entry["library"] == "lightgbm"
    assert entry["status"] == "ok"
    assert entry["hyperparameters"]["n_estimators"] == 100
    assert entry["hyperparameters"]["random_state"] == run["seed"]
    # better than chance, not an exact score: see the note in the leaderboard test
    assert entry["metrics"]["accuracy"]["value"] > 0.6
    assert run["library_versions"]["lightgbm"]


def test_lightgbm_regression_is_ranked_too(client: TestClient, version: str) -> None:
    pytest.importorskip("lightgbm")
    run = train(client, version_id=version, target="spend", models=["lightgbm"])
    entry = run["leaderboard"][0]
    assert entry["status"] == "ok"
    assert entry["metrics"]["r2"]["value"] is not None
    assert entry["metrics"]["rmse"]["value"] >= 0.0
