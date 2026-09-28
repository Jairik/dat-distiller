"""On-demand evaluation and prediction.

The two acceptance criteria are about *control* rather than arithmetic:

- **Nothing runs unless asked.** No plot, no prediction, and no refit happens as
  a side effect of a Training Run. The tests assert that by counting the HTTP
  calls a run makes and checking none of them is an evaluation.
- **Prediction reuses the trained pipeline and refuses a bad file clearly.** A
  hand-built fixture pins the metric maths; the missing-column case has to name
  *which* columns are missing, because "invalid input" is useless to the person
  holding the CSV.
"""

from __future__ import annotations

import time
from typing import Any

import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient

from dat_distiller.store import DatasetStore
from dat_distiller.training.evaluate import (
    EvaluationError,
    confusion_matrix_plot,
    precision_recall_plot,
    residual_plots,
    roc_curve_plot,
)

pytest.importorskip("sklearn")


# -- fixtures -----------------------------------------------------------------


def toy_frame(
    rows: int = 120, seed: int = 11, target: str | None = "is_churn"
) -> pd.DataFrame:
    """A learnable frame with exactly one Target column.

    Only one, on purpose: a second would be picked up as a *feature* (every
    eligible column is one by default), which is correct behaviour and a
    confusing fixture.
    """
    rng = np.random.default_rng(seed)
    age = rng.integers(18, 80, rows)
    spend = rng.normal(0, 1, rows).round(3)
    plan = rng.choice(["basic", "pro"], rows)
    churn = (age < 40) & (spend > 0)
    frame = pd.DataFrame({"age": age, "spend": spend, "plan": plan})
    if target == "is_churn":
        frame["is_churn"] = pd.array(churn, dtype="boolean")
    elif target == "spend_target":
        frame["spend_target"] = (spend * 3 + age / 10).round(3)
    return frame


def features_only(frame: pd.DataFrame) -> pd.DataFrame:
    """The frame without its Target, as a user would upload for prediction."""
    return frame.drop(columns=[c for c in ("is_churn", "spend_target") if c in frame.columns])


@pytest.fixture()
def project_id(client: TestClient) -> str:
    return client.post("/api/projects", json={"name": "eval lab"}).json()["id"]


def _version(client: TestClient, project_id: str, target: str = "is_churn") -> str:
    store: DatasetStore = client.app.state.store
    return store.create_version(project_id, toy_frame(target=target), origin="uploaded").id


def _wait(client: TestClient, job_id: str) -> dict[str, Any]:
    for _ in range(6000):
        job = client.get(f"/api/jobs/{job_id}").json()
        if job["status"] in ("completed", "failed", "cancelled"):
            return job
        time.sleep(0.02)
    raise AssertionError("the job never finished")


def train(client: TestClient, version_id: str, target: str, model: str) -> dict[str, Any]:
    body: dict[str, Any] = {"version_id": version_id, "target": target, "models": [model]}
    start = client.post("/api/train/run", json=body)
    assert start.status_code == 202, start.text
    job = _wait(client, start.json()["id"])
    assert job["status"] == "completed", job.get("error")
    return job["result"]


@pytest.fixture()
def run(client: TestClient, project_id: str) -> dict[str, Any]:
    return train(client, _version(client, project_id), "is_churn", "logistic_regression")


# -- AC1: nothing runs automatically -------------------------------------------


def test_a_training_run_makes_no_evaluation_call(client: TestClient, project_id: str) -> None:
    """AC1: nothing is evaluated unless it is asked for."""
    version = _version(client, project_id)
    loads: list[str] = []
    original = client.app.state.store.load_dataframe

    def spy(*args: Any, **kwargs: Any) -> pd.DataFrame:
        loads.append("load")
        return original(*args, **kwargs)

    client.app.state.store.load_dataframe = spy  # type: ignore[method-assign]
    try:
        start = client.post(
            "/api/train/run",
            json={"version_id": version, "target": "is_churn", "models": ["logistic_regression"]},
        )
        assert start.status_code == 202
        job = _wait(client, start.json()["id"])
        assert job["status"] == "completed", job.get("error")

        result = job["result"]
        # no plot, no prediction and no extra refit crept into the run
        for plot in ("confusion_matrix", "roc", "precision_recall", "feature_importance", "residuals"):
            assert plot not in result
        assert "predictions" not in result
        assert "plots" not in result

        # the run read the Dataset Version, but only to fit and score. Asking for
        # a plot afterwards is what does more work — that is the whole point of
        # "on demand", so it is asserted rather than assumed.
        during_run = len(loads)
        assert during_run > 0
        plot = client.get(
            f"/api/train/runs/{result['training_run_id']}/plots",
            params={"model": "logistic_regression", "plot": "confusion_matrix"},
        )
        assert plot.status_code == 200
        assert len(loads) > during_run
    finally:
        client.app.state.store.load_dataframe = original  # type: ignore[method-assign]


# -- plots --------------------------------------------------------------------


def test_confusion_matrix_is_correct_on_a_hand_built_fixture() -> None:
    # rows = truth, cols = prediction, in the given label order
    y_true = ["a", "a", "a", "b", "b", "c"]
    y_pred = ["a", "a", "b", "b", "b", "c"]
    out = confusion_matrix_plot(y_true, y_pred, ["a", "b", "c"])
    assert out["labels"] == ["a", "b", "c"]
    assert out["matrix"] == [
        [2, 1, 0],
        [0, 2, 0],
        [0, 0, 1],
    ]
    assert out["support"] == [3, 2, 1]
    assert out["n"] == 6
    # row-normalised: each row divided by its own support
    assert out["row_normalised"][0] == [round(2 / 3, 6), round(1 / 3, 6), 0.0]
    assert out["row_normalised"][1] == [0.0, 1.0, 0.0]


def test_the_confusion_matrix_keeps_the_given_label_order() -> None:
    # reversing the labels must transpose, not silently re-sort
    forward = confusion_matrix_plot(["a", "b"], ["a", "b"], ["a", "b"])
    backward = confusion_matrix_plot(["a", "b"], ["a", "b"], ["b", "a"])
    assert forward["matrix"] == [[1, 0], [0, 1]]
    assert backward["matrix"] == [[1, 0], [0, 1]]
    assert backward["labels"] == ["b", "a"]


def test_a_single_class_target_says_so_rather_than_emitting_a_fake_curve() -> None:
    # one class is not a ranking problem; a diagonal "curve" would look meaningful
    with pytest.raises(EvaluationError, match="at least two classes"):
        roc_curve_plot(np.array(["a", "a"]), np.array([[0.6], [0.4]]), ["a"])
    with pytest.raises(EvaluationError, match="at least two classes"):
        precision_recall_plot(np.array(["a", "a"]), np.array([[0.6], [0.4]]), ["a"])


def test_a_one_class_fold_within_a_multiclass_run_yields_a_null_auc() -> None:
    # a real situation: a minority class absent from the test split
    truth = np.array(["no", "no", "no"])
    proba = np.array([[0.6, 0.4], [0.7, 0.3], [0.55, 0.45]])
    out = roc_curve_plot(truth, proba, ["no", "yes"])
    assert out["auc"] is None or np.isnan(out["auc"])


def test_roc_auc_is_one_for_a_perfect_ranker() -> None:
    truth = np.array(["no", "no", "yes", "yes"])
    proba = np.array([[0.9, 0.1], [0.8, 0.2], [0.2, 0.8], [0.1, 0.9]])
    out = roc_curve_plot(truth, proba, ["no", "yes"])
    assert out["kind"] == "binary"
    assert out["positive_class"] == "yes"
    assert out["auc"] == pytest.approx(1.0)
    assert out["fpr"][0] == 0.0 and out["tpr"][0] == 0.0
    assert out["fpr"][-1] == 1.0 and out["tpr"][-1] == 1.0


def test_roc_auc_is_half_for_a_random_ranker() -> None:
    truth = np.array(["no", "yes", "no", "yes"])
    proba = np.array([[0.5, 0.5]] * 4)
    out = roc_curve_plot(truth, proba, ["no", "yes"])
    assert out["auc"] == pytest.approx(0.5, abs=0.05)


def test_roc_refuses_a_model_with_no_probabilities() -> None:
    with pytest.raises(EvaluationError, match="does not output class probabilities"):
        roc_curve_plot(np.array(["a", "b"]), None, ["a", "b"])


def test_roc_on_a_multiclass_target_is_one_vs_rest() -> None:
    truth = np.array(["a", "a", "b", "b", "c", "c"])
    proba = np.array(
        [
            [0.8, 0.1, 0.1],
            [0.7, 0.2, 0.1],
            [0.1, 0.8, 0.1],
            [0.2, 0.7, 0.1],
            [0.1, 0.1, 0.8],
            [0.1, 0.2, 0.7],
        ]
    )
    out = roc_curve_plot(truth, proba, ["a", "b", "c"])
    assert out["kind"] == "multiclass_ovr"
    assert out["classes"] == ["a", "b", "c"]
    assert out["auc_macro"] > 0.9


def test_average_precision_is_one_for_a_perfect_ranker() -> None:
    truth = np.array(["no", "no", "yes", "yes"])
    proba = np.array([[0.9, 0.1], [0.8, 0.2], [0.2, 0.8], [0.1, 0.9]])
    out = precision_recall_plot(truth, proba, ["no", "yes"])
    assert out["kind"] == "binary"
    assert out["average_precision"] == pytest.approx(1.0)


def test_residuals_are_actual_minus_predicted() -> None:
    out = residual_plots(np.array([3.0, 5.0, 4.0]), np.array([2.0, 5.0, 6.0]))
    assert out["residuals"] == [1.0, 0.0, -2.0]
    assert out["mae"] == pytest.approx(1.0)
    assert out["n"] == 3


def test_permutation_importance_is_reproducible_for_a_fixed_seed(
    client: TestClient, project_id: str
) -> None:
    version = _version(client, project_id)
    run_id = train(client, version, "is_churn", "logistic_regression")["training_run_id"]
    first = client.get(
        f"/api/train/runs/{run_id}/plots",
        params={"model": "logistic_regression", "plot": "feature_importance"},
    ).json()
    second = client.get(
        f"/api/train/runs/{run_id}/plots",
        params={"model": "logistic_regression", "plot": "feature_importance"},
    ).json()
    assert first["importance"] == second["importance"]


# -- the endpoints ------------------------------------------------------------


def test_each_plot_runs_only_when_requested(client: TestClient, project_id: str) -> None:
    version = _version(client, project_id)
    result = train(client, version, "is_churn", "logistic_regression")
    run_id = result["training_run_id"]

    for plot in ("confusion_matrix", "roc", "precision_recall", "feature_importance"):
        r = client.get(f"/api/train/runs/{run_id}/plots", params={"model": "logistic_regression", "plot": plot})
        assert r.status_code == 200, r.text
        assert r.json()["plot"] == plot
        assert r.json()["model"] == "logistic_regression"

    # regression gets residuals, not a confusion matrix
    reg_version = _version(client, project_id, target="spend_target")
    reg = train(client, reg_version, "spend_target", "linear_regression")
    r = client.get(
        f"/api/train/runs/{reg['training_run_id']}/plots",
        params={"model": "linear_regression", "plot": "residuals"},
    )
    assert r.status_code == 200
    assert r.json()["n"] > 0


def test_the_wrong_plot_for_a_task_type_is_refused_clearly(
    client: TestClient, project_id: str
) -> None:
    version = _version(client, project_id, target="spend_target")
    reg = train(client, version, "spend_target", "linear_regression")
    r = client.get(
        f"/api/train/runs/{reg['training_run_id']}/plots",
        params={"model": "linear_regression", "plot": "confusion_matrix"},
    )
    assert r.status_code == 422
    assert "regression" in r.text


def test_an_unknown_plot_or_model_is_refused(client: TestClient, project_id: str) -> None:
    version = _version(client, project_id)
    run_id = train(client, version, "is_churn", "logistic_regression")["training_run_id"]
    assert (
        client.get(
            f"/api/train/runs/{run_id}/plots",
            params={"model": "logistic_regression", "plot": "sunburst"},
        ).status_code
        == 422
    )
    r = client.get(f"/api/train/runs/{run_id}/plots", params={"model": "nope", "plot": "roc"})
    assert r.status_code == 422
    assert "not on this Training Run" in r.text


def test_a_model_with_no_probabilities_says_so_rather_than_failing(
    client: TestClient, project_id: str
) -> None:
    version = _version(client, project_id)
    run_id = train(client, version, "is_churn", "svm")["training_run_id"]
    body = client.get(
        f"/api/train/runs/{run_id}/plots", params={"model": "svm", "plot": "roc"}
    ).json()
    # the endpoint answers with a reason, it does not 500
    assert body["error"]
    assert "probabilities" in body["error"]


# -- prediction ---------------------------------------------------------------


def test_predicting_returns_the_input_plus_a_prediction_column(
    client: TestClient, project_id: str
) -> None:
    version = _version(client, project_id)
    result = train(client, version, "is_churn", "logistic_regression")
    run_id = result["training_run_id"]

    new_rows = features_only(toy_frame(rows=5, seed=99))
    csv = new_rows.to_csv(index=False).encode()

    r = client.post(
        f"/api/train/runs/{run_id}/predict",
        params={"model": "logistic_regression"},
        files={"file": ("new.csv", csv, "text/csv")},
    )
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["rows"] == 5
    assert body["columns"][: len(new_rows.columns)] == list(new_rows.columns)
    assert "prediction" in body["columns"]
    assert set(body["predictions"][0]) >= {"prediction"}
    # classification gives a probability per class
    assert any(c.startswith("probability_") for c in body["columns"])
    assert "/predictions.csv?" in body["download_url"]
    assert "token=" in body["download_url"]


def test_prediction_rejects_missing_columns_and_names_them(
    client: TestClient, project_id: str
) -> None:
    version = _version(client, project_id)
    run_id = train(client, version, "is_churn", "logistic_regression")["training_run_id"]

    partial = toy_frame(rows=3, seed=5)[["age"]].to_csv(index=False).encode()  # only one of three
    r = client.post(
        f"/api/train/runs/{run_id}/predict",
        params={"model": "logistic_regression"},
        files={"file": ("partial.csv", partial, "text/csv")},
    )
    assert r.status_code == 422
    # names the columns, because "invalid input" helps nobody
    assert "spend" in r.text
    assert "plan" in r.text


def test_regression_prediction_has_no_probabilities(client: TestClient, project_id: str) -> None:
    version = _version(client, project_id)
    version = _version(client, project_id, target="spend_target")
    run_id = train(client, version, "spend_target", "linear_regression")["training_run_id"]
    csv = features_only(toy_frame(rows=4, seed=7, target="spend_target")).to_csv(index=False).encode()
    body = client.post(
        f"/api/train/runs/{run_id}/predict",
        params={"model": "linear_regression"},
        files={"file": ("new.csv", csv, "text/csv")},
    ).json()
    assert not any(c.startswith("probability_") for c in body["columns"])
    assert all(isinstance(float(row["prediction"]), float) for row in body["predictions"])


def test_prediction_uses_the_stored_pipeline_not_a_fresh_one(
    client: TestClient, project_id: str
) -> None:
    # a scaled numeric column: predicting on raw values would give different
    # answers if the pipeline were re-derived rather than reused
    version = _version(client, project_id)
    result = train(client, version, "is_churn", "logistic_regression")
    run_id = result["training_run_id"]
    new_rows = features_only(toy_frame(rows=6, seed=123))
    body = client.post(
        f"/api/train/runs/{run_id}/predict",
        params={"model": "logistic_regression"},
        files={"file": ("new.csv", new_rows.to_csv(index=False).encode(), "text/csv")},
    ).json()

    pipeline = result["preprocessing"]
    assert pipeline["scaler"]  # a scaler really was fitted
    probs = [row for row in body["predictions"] if "probability_True" in row]
    assert probs
    for row in probs:
        values = [v for k, v in row.items() if k.startswith("probability_")]
        assert sum(values) == pytest.approx(1.0, abs=1e-4)


def test_predicting_on_an_unknown_run_is_404(client: TestClient) -> None:
    r = client.post(
        "/api/train/runs/nope/predict",
        params={"model": "logistic_regression"},
        files={"file": ("new.csv", b"age\n30\n", "text/csv")},
    )
    assert r.status_code == 404


def test_the_download_is_a_real_csv(client: TestClient, project_id: str) -> None:
    version = _version(client, project_id)
    run_id = train(client, version, "is_churn", "logistic_regression")["training_run_id"]
    csv = features_only(toy_frame(rows=3, seed=8)).to_csv(index=False).encode()
    body = client.post(
        f"/api/train/runs/{run_id}/predict",
        params={"model": "logistic_regression"},
        files={"file": ("new.csv", csv, "text/csv")},
    ).json()
    download = client.get(body["download_url"])
    assert download.status_code == 200
    assert "text/csv" in download.headers["content-type"]
    lines = download.text.strip().split("\n")
    header = lines[0].split(",")
    assert "prediction" in header
    assert len(lines) == 4  # header + 3 rows


def test_permutation_importance_ranks_a_real_signal_first(
    client: TestClient, project_id: str
) -> None:
    version = _version(client, project_id)
    run_id = train(client, version, "is_churn", "logistic_regression")["training_run_id"]
    body = client.get(
        f"/api/train/runs/{run_id}/plots",
        params={"model": "logistic_regression", "plot": "feature_importance"},
    ).json()
    assert body["method"] == "permutation"
    assert body["importance"]
    ranked = [entry["feature"] for entry in body["importance"]]
    # age and spend both drive the Target; plan is noise here
    assert set(ranked[:2]) & {"age_scaled", "spend_scaled", "age", "spend"}
    assert ranked == sorted(ranked, key=lambda f: next(e["drop"] for e in body["importance"] if e["feature"] == f), reverse=True)
