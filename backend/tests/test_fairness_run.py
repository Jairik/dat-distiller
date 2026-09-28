"""The Fairness Report against a real Training Run: reuse, refit, refusals, Checks.

These tests really train a Model and then really refit it, so they are guarded
with ``pytest.importorskip("sklearn")`` — a bare install stays green. What they
pin down is the part the hand-built fixture in ``test_fairness.py`` cannot:

- the report reuses the run's **own** split and preprocessing, verified against
  the indices the run recorded, so it measures the same rows as the leaderboard;
- a Model that could not be refitted is refused with a readable reason instead of
  a report of made-up numbers;
- gaps over the threshold become warning Checks against the Training Run, with
  the groups and the numbers in the message, and they do not pile up on a repeat;
- nothing runs at training time, and no Check is raised by a Training Run that was
  never asked for a report.
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
from dat_distiller.training.checks import value_key
from dat_distiller.training.fairness import (
    DEFAULT_GAP_THRESHOLD,
    FAIRNESS_GAP_CHECK_KIND,
    FAIRNESS_USED_CHECK_KIND,
    FairnessError,
    encode_with_classes,
    fairness_report,
    load_run_context,
    positive_class_code,
    register_fairness_checks,
)

pytest.importorskip("sklearn")


# -- data --------------------------------------------------------------------


def labeled_frame(rows: int = 60, seed: int = 5) -> pd.DataFrame:
    """A learnable frame with a Sensitive Attribute that is genuinely unequal.

    `is_spam` is a real function of two features, and the base rate differs by
    region, so a Model that simply predicts the majority for everybody still shows
    a demographic parity gap — which is what makes the fixture a real test rather
    than a formality.
    """
    rng = np.random.default_rng(seed)
    age = rng.integers(18, 80, rows)
    spend = rng.normal(0, 1, rows).round(3)
    region = rng.choice(["north", "south", "east"], rows, p=[0.5, 0.3, 0.2])
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


def unfair_frame(rows: int = 60, seed: int = 9) -> pd.DataFrame:
    """A frame where the Sensitive Attribute *is* the answer.

    `is_spam` is exactly `region == 'north'`, and the features carry no signal at
    all, so a Model fitted on age and spend has nothing to learn and falls back on
    the majority class. That produces a large, reproducible selection-rate gap
    between the groups — which is what the Check tests need. Splitting is
    stratified on the Target, so every group keeps held-out rows either way.
    """
    rng = np.random.default_rng(seed)
    region = rng.choice(["north", "south", "east"], rows, p=[0.6, 0.25, 0.15])
    return pd.DataFrame(
        {
            "age": rng.integers(18, 80, rows),
            "spend": rng.normal(0, 1, rows).round(3),
            "region": region,
            "is_spam": pd.array(region == "north", dtype="boolean"),
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
    return client.post("/api/projects", json={"name": "fairness lab"}).json()["id"]


@pytest.fixture
def version(client: TestClient, project_id: str) -> str:
    store: DatasetStore = client.app.state.store
    return store.create_version(project_id, labeled_frame(), origin="uploaded").id


@pytest.fixture
def unfair_version(client: TestClient, project_id: str) -> str:
    store: DatasetStore = client.app.state.store
    return store.create_version(project_id, unfair_frame(), origin="uploaded").id


def wait_for_job(client: TestClient, job_id: str, timeout: float = 240.0) -> dict[str, Any]:
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


def run_id_of(
    client: TestClient,
    version: str,
    target: str = "is_spam",
    models: list[str] | None = None,
    **body: Any,
) -> str:
    return train(
        client,
        version_id=version,
        target=target,
        models=models if models is not None else ["knn", "random_forest"],
        **body,
    )["training_run_id"]


def report(client: TestClient, run_id: str, **body: Any) -> dict[str, Any]:
    response = client.post(f"/api/train/runs/{run_id}/fairness", json=body)
    assert response.status_code == 200, response.text
    return response.json()


# -- the endpoint ------------------------------------------------------------


def test_the_endpoint_produces_per_group_metrics_for_a_classification_run(
    client: TestClient, version: str
) -> None:
    run_id = run_id_of(client, version)
    body = report(client, run_id, sensitive_attribute="region")

    assert body["training_run_id"] == run_id
    assert body["task_type"] == "classification"
    assert body["target"] == "is_spam"
    assert body["model"] in {"knn", "random_forest"}
    assert body["model_label"]
    assert body["sensitive_attribute"] == "region"
    assert body["threshold"] == DEFAULT_GAP_THRESHOLD
    assert body["threshold_source"] == "settings", "the threshold comes from settings"
    assert body["refit"] is True
    assert body["positive_class"] == "bool:True"
    assert body["classes"] == ["bool:False", "bool:True"]

    # The held-out rows are the run's own, and every group is accounted for.
    run = client.get(f"/api/train/runs/{run_id}").json()
    assert body["split"]["test_rows"] == run["test_split"]["rows"] == 12
    assert body["split"]["train_rows"] == run["training_split"]["n_samples"]
    assert body["split"]["reused_stored_split"] is True
    assert body["overall"]["n_test"] == 12
    assert sum(group["n_test"] for group in body["groups"]) == 12

    groups = {entry["group"]: entry for entry in body["groups"]}
    assert set(groups) >= {"str:north", "str:south", "str:east"}
    for name in ("accuracy", "tpr", "fpr", "selection_rate"):
        assert name in groups["str:north"]["metrics"], name
        entry = groups["str:north"]["metrics"][name]
        # 12 held-out rows over three groups is a small split, so a group can
        # genuinely hold no positive (or no negative) case. Then the metric is
        # null with a reason, which is what this asserts rather than a number.
        if entry["value"] is None:
            assert entry["reason"], name
        else:
            assert 0.0 <= entry["value"] <= 1.0, name
    # The per-group accuracies are what the overall accuracy is made of: every
    # held-out row belongs to exactly one group, so the weighted average of the
    # groups' accuracies is the run's own accuracy. (TPR and FPR are not additive
    # this way, which is why accuracy is the one checked here.)
    with_accuracy = [
        g for g in groups.values() if g["metrics"]["accuracy"]["value"] is not None
    ]
    total = sum(g["n_test"] for g in with_accuracy)
    assert total == 12
    assert sum(
        g["n_test"] * g["metrics"]["accuracy"]["value"] for g in with_accuracy
    ) / total == pytest.approx(body["overall"]["accuracy"]["value"], abs=1e-6)


def test_the_report_carries_every_gap_with_its_groups_and_numbers(
    client: TestClient, version: str
) -> None:
    body = report(client, run_id_of(client, version), sensitive_attribute="region")
    names = [gap["name"] for gap in body["gaps"]]
    assert names == ["demographic_parity", "tpr", "fpr", "accuracy"]
    assert body["largest_gap"] is not None
    for gap in body["gaps"]:
        assert gap["threshold"] == DEFAULT_GAP_THRESHOLD
        assert gap["groups_total"] == body["n_groups"]
        if gap["value"] is None:
            assert gap["reason"], gap["name"]
            assert gap["exceeds"] is None
        else:
            assert gap["value"] == pytest.approx(
                gap["highest_value"] - gap["lowest_value"], abs=0.01
            )
            assert gap["highest_group"] and gap["lowest_group"]
            assert gap["exceeds"] == (gap["value"] > DEFAULT_GAP_THRESHOLD)
    # `gaps_exceeding_threshold` is exactly the flagged set, ordered widest
    # first — a different order from `gaps`, which is in declaration order.
    flagged = {gap["name"]: gap for gap in body["gaps"] if gap.get("exceeds") is True}
    assert set(body["gaps_exceeding_threshold"]) == set(flagged)
    values = [flagged[name]["value"] for name in body["gaps_exceeding_threshold"]]
    assert values == sorted(values, reverse=True)
    assert body["largest_gap"] in body["gaps"]


def test_a_regression_run_reports_per_group_mae(client: TestClient, version: str) -> None:
    run_id = run_id_of(client, version, target="spend", models=["knn", "linear_regression"])
    body = report(client, run_id, sensitive_attribute="region")
    assert body["task_type"] == "regression"
    assert body["target"] == "spend"
    assert body["positive_class"] is None
    groups = {entry["group"]: entry for entry in body["groups"]}
    assert groups
    for entry in groups.values():
        if entry["measured"]:
            assert entry["metrics"]["mae"]["value"] >= 0.0
            assert "rmse" in entry["metrics"]
    mae_gap = next(gap for gap in body["gaps"] if gap["name"] == "mae")
    assert mae_gap["unit"] == "target_units"
    assert mae_gap["comparable"] is False
    assert mae_gap["exceeds"] is None, "a gap in Target units is not a rate"
    relative = next(gap for gap in body["gaps"] if gap["name"] == "mae_relative")
    assert relative["unit"] == "rate"
    assert relative["comparable"] is True
    if relative["value"] is not None:
        assert relative["exceeds"] == (relative["value"] > DEFAULT_GAP_THRESHOLD)


# -- it reuses the run's own pipeline ----------------------------------------


def test_the_report_measures_the_runs_own_held_out_rows(
    client: TestClient, version: str
) -> None:
    """The load-bearing test: the report predicts on the run's stored indices, so
    a gap it reports is a gap on the rows the leaderboard scored."""
    run_id = run_id_of(client, version)
    run = client.get(f"/api/train/runs/{run_id}").json()
    store: DatasetStore = client.app.state.store
    job = client.app.state.jobs.get(run_id)
    context = load_run_context(store, job)
    assert context.test_indices == run["test_split"]["indices"]
    assert context.train_indices == run["training_split"]["rows"]
    assert context.preprocessing == run["preprocessing"]
    assert context.setup.feature_columns == run["setup"]["feature_columns"]

    # The test matrix is built with the run's stored spec, so the Model sees
    # exactly what it was fitted on: same width, same column order.
    matrix = context.test_matrix()
    assert matrix.shape[1] == run["n_features"]
    assert len(matrix) == run["test_split"]["rows"]

    # And the report's per-group row counts match the run's own held-out rows.
    body = report(client, run_id, sensitive_attribute="region")
    assert body["split"]["test_rows"] == len(context.test_indices)
    assert sum(g["n_test"] for g in body["groups"]) == len(context.test_indices)


def test_a_run_whose_stored_split_cannot_be_reproduced_is_refused(
    client: TestClient, version: str
) -> None:
    """If the stored split and the request disagree, the report is refused rather
    than computed on rows the leaderboard never scored."""
    run_id = run_id_of(client, version)
    with client.app.state.store.db.connect() as conn:
        row = conn.execute("SELECT checkpoint_json FROM jobs WHERE id = ?", (run_id,)).fetchone()
        checkpoint = json.loads(row["checkpoint_json"])
        checkpoint["run"]["test_split"]["indices"] = [0, 1, 2]
        conn.execute(
            "UPDATE jobs SET checkpoint_json = ? WHERE id = ?",
            (json.dumps(checkpoint), run_id),
        )
    response = client.post(
        f"/api/train/runs/{run_id}/fairness", json={"sensitive_attribute": "region"}
    )
    assert response.status_code == 422
    detail = response.json()["detail"]
    assert "held-out rows cannot be reproduced" in detail
    assert "re-run" in detail


def test_a_run_with_no_stored_held_out_rows_is_refused(
    client: TestClient, version: str
) -> None:
    run_id = run_id_of(client, version)
    with client.app.state.store.db.connect() as conn:
        row = conn.execute("SELECT checkpoint_json FROM jobs WHERE id = ?", (run_id,)).fetchone()
        checkpoint = json.loads(row["checkpoint_json"])
        checkpoint["run"]["test_split"]["indices"] = []
        conn.execute(
            "UPDATE jobs SET checkpoint_json = ? WHERE id = ?",
            (json.dumps(checkpoint), run_id),
        )
    response = client.post(
        f"/api/train/runs/{run_id}/fairness", json={"sensitive_attribute": "region"}
    )
    assert response.status_code == 422
    assert "no held-out rows" in response.json()["detail"]


def test_a_run_with_no_stored_preprocessing_is_refused(
    client: TestClient, version: str
) -> None:
    run_id = run_id_of(client, version)
    with client.app.state.store.db.connect() as conn:
        row = conn.execute("SELECT checkpoint_json FROM jobs WHERE id = ?", (run_id,)).fetchone()
        checkpoint = json.loads(row["checkpoint_json"])
        checkpoint["run"]["preprocessing"] = {}
        conn.execute(
            "UPDATE jobs SET checkpoint_json = ? WHERE id = ?",
            (json.dumps(checkpoint), run_id),
        )
    response = client.post(
        f"/api/train/runs/{run_id}/fairness", json={"sensitive_attribute": "region"}
    )
    assert response.status_code == 422
    assert "no preprocessing pipeline" in response.json()["detail"]


def test_the_refit_reproduces_the_models_own_held_out_accuracy(
    client: TestClient, version: str
) -> None:
    """The refit is deterministic from the run's own data and hyperparameters, so
    the per-group numbers come from the same Model the leaderboard scored — the
    group's rows weighted by their accuracy reproduce the run's own accuracy."""
    run_id = run_id_of(client, version)
    run = client.get(f"/api/train/runs/{run_id}").json()
    body = report(client, run_id, sensitive_attribute="region", model="knn")
    entry = next(e for e in run["leaderboard"] if e["model"] == "knn")
    assert body["model"] == "knn"
    assert body["hyperparameters"] == entry["hyperparameters"]
    assert body["seed"] == run["seed"]
    assert body["overall"]["accuracy"]["value"] == pytest.approx(
        entry["metrics"]["accuracy"]["value"], abs=0.02
    )


# -- refusals ----------------------------------------------------------------


def test_an_unknown_model_is_refused_and_says_what_is_available(
    client: TestClient, version: str
) -> None:
    run_id = run_id_of(client, version)
    response = client.post(
        f"/api/train/runs/{run_id}/fairness",
        json={"sensitive_attribute": "region", "model": "catboost"},
    )
    assert response.status_code == 422
    detail = response.json()["detail"]
    assert "not on this Training Run's leaderboard" in detail
    assert "knn" in detail and "random_forest" in detail


def test_a_model_that_failed_to_fit_is_refused_with_its_error(
    client: TestClient, version: str
) -> None:
    run_id = run_id_of(
        client,
        version,
        models=["knn", "random_forest"],
        hyperparameters={"knn": {"n_neighbors": "three"}},
    )
    response = client.post(
        f"/api/train/runs/{run_id}/fairness",
        json={"sensitive_attribute": "region", "model": "knn"},
    )
    assert response.status_code == 422
    assert "failed to fit" in response.json()["detail"]
    # The Model that did fit still gets a report.
    assert report(client, run_id, sensitive_attribute="region", model="random_forest")["model"] == (
        "random_forest"
    )


def test_an_unknown_or_impossible_sensitive_attribute_is_refused(
    client: TestClient, version: str
) -> None:
    run_id = run_id_of(client, version)
    unknown = client.post(
        f"/api/train/runs/{run_id}/fairness", json={"sensitive_attribute": "nope"}
    )
    assert unknown.status_code == 422
    assert "unknown Sensitive Attribute" in unknown.json()["detail"]
    the_target = client.post(
        f"/api/train/runs/{run_id}/fairness", json={"sensitive_attribute": "is_spam"}
    )
    assert the_target.status_code == 422
    assert "cannot also be the Target" in the_target.json()["detail"]
    empty = client.post(
        f"/api/train/runs/{run_id}/fairness", json={"sensitive_attribute": "   "}
    )
    assert empty.status_code == 422
    assert "name a Sensitive Attribute" in empty.json()["detail"]


def test_a_run_that_does_not_exist_or_is_not_a_training_run_is_a_404(
    client: TestClient, version: str
) -> None:
    assert (
        client.post(
            "/api/train/runs/nope/fairness", json={"sensitive_attribute": "region"}
        ).status_code
        == 404
    )
    setup = client.post("/api/train/setup", json={"version_id": version, "target": "is_spam"})
    setup_job = wait_for_job(client, setup.json()["id"])
    assert setup_job["status"] == "completed"
    response = client.post(
        f"/api/train/runs/{setup_job['id']}/fairness", json={"sensitive_attribute": "region"}
    )
    assert response.status_code == 404
    assert response.json()["detail"] == "not a training run"


def test_a_run_still_in_flight_is_a_409(client: TestClient, version: str) -> None:
    """A report needs a finished run, because it re-measures the Model it fitted."""
    start = client.post(
        "/api/train/run", json={"version_id": version, "target": "is_spam", "models": ["knn"]}
    )
    job_id = start.json()["id"]
    wait_for_job(client, job_id)
    with client.app.state.store.db.connect() as conn:
        conn.execute("UPDATE jobs SET status = 'running' WHERE id = ?", (job_id,))
    response = client.post(
        f"/api/train/runs/{job_id}/fairness", json={"sensitive_attribute": "region"}
    )
    assert response.status_code == 409
    assert "needs a finished run" in response.json()["detail"]


def test_an_impossible_threshold_is_refused_by_the_schema_and_the_module(
    client: TestClient, version: str
) -> None:
    run_id = run_id_of(client, version)
    assert (
        client.post(
            f"/api/train/runs/{run_id}/fairness",
            json={"sensitive_attribute": "region", "gap_threshold": 1.5},
        ).status_code
        == 422
    )
    assert (
        client.post(
            f"/api/train/runs/{run_id}/fairness",
            json={"sensitive_attribute": "region", "gap_threshold": -0.2},
        ).status_code
        == 422
    )
    with pytest.raises(FairnessError, match="between 0 and 1"):
        fairness_report(
            store=client.app.state.store,
            job=client.app.state.jobs.get(run_id),
            sensitive_attribute="region",
            threshold=2.0,
        )


def test_a_positive_label_that_is_not_a_class_is_refused(
    client: TestClient, version: str
) -> None:
    run_id = run_id_of(client, version)
    response = client.post(
        f"/api/train/runs/{run_id}/fairness",
        json={"sensitive_attribute": "region", "positive_label": "maybe"},
    )
    assert response.status_code == 422
    assert "is not a class of this Target" in response.json()["detail"]


def test_a_named_positive_class_changes_which_rates_are_called_positive(
    client: TestClient, version: str
) -> None:
    run_id = run_id_of(client, version)
    default = report(client, run_id, sensitive_attribute="region")
    flipped = report(client, run_id, sensitive_attribute="region", positive_label="False")
    assert default["positive_class"] == "bool:True"
    assert flipped["positive_class"] == "bool:False"
    assert flipped["positive_label"] == "False"
    # Same rows, same Model, different notion of positive -> different rates.
    assert flipped["groups"][0]["n_test"] == default["groups"][0]["n_test"]
    north_default = next(g for g in default["groups"] if g["group"] == "str:north")
    north_flipped = next(g for g in flipped["groups"] if g["group"] == "str:north")
    assert north_default["n_actual_positive"] + north_default["n_actual_negative"] == (
        north_flipped["n_actual_positive"] + north_flipped["n_actual_negative"]
    )


# -- the Sensitive Attribute as a feature -------------------------------------


def test_an_excluded_sensitive_attribute_says_so_and_an_included_one_says_more(
    client: TestClient, project_id: str, version: str
) -> None:
    """#31 excludes the Sensitive Attribute from the features by default. A
    report on a column the Model *could* see is a different finding, and the
    report has to say which of the two this is."""
    store: DatasetStore = client.app.state.store

    # The run declares 'region' and does not include it: the default.
    excluded_id = run_id_of(client, version, sensitive_attribute="region")
    excluded = report(client, excluded_id, sensitive_attribute="region")
    assert excluded["sensitive_attribute_excluded_from_features"] is True
    assert excluded["declared_sensitive_attribute"] == "region"
    assert not any("was a feature" in note for note in excluded["notes"])
    # An excluded attribute earns no "you let the Model see it" note at all.
    assert all(
        check["kind"] != FAIRNESS_USED_CHECK_KIND
        for check in excluded["checks_raised"]
    )

    # A run that explicitly included it: the report must flag that.
    frame = labeled_frame(60)
    included_version = store.create_version(project_id, frame, origin="uploaded").id
    included_id = run_id_of(
        client,
        included_version,
        sensitive_attribute="region",
        include_sensitive_attribute=True,
    )
    included = report(client, included_id, sensitive_attribute="region")
    assert included["sensitive_attribute_excluded_from_features"] is False
    assert any("was a feature of this Model" in note for note in included["notes"])
    used = [
        check
        for check in included["checks_raised"]
        if check["kind"] == FAIRNESS_USED_CHECK_KIND
    ]
    assert len(used) == 1
    assert used[0]["severity"] == "info"
    assert "was a feature" in used[0]["message"]
    assert used[0]["details"]["sensitive_attribute"] == "region"
    assert used[0]["details"]["sensitive_attribute_excluded_from_features"] is False


def test_reporting_an_attribute_the_run_treated_as_a_feature_is_called_out(
    client: TestClient, version: str
) -> None:
    """'age' is a plain feature of the run; asking for a report on it must not be
    silently answered as if it were the declared Sensitive Attribute."""
    run_id = run_id_of(client, version, sensitive_attribute="region")
    body = report(client, run_id, sensitive_attribute="age")
    assert body["sensitive_attribute"] == "age"
    assert body["declared_sensitive_attribute"] == "region"
    assert body["sensitive_attribute_excluded_from_features"] is False
    assert any("declared 'region'" in note for note in body["notes"])


# -- the Checks --------------------------------------------------------------


def test_a_gap_over_the_threshold_raises_a_warning_check_naming_the_groups(
    client: TestClient, unfair_version: str
) -> None:
    """The frame is one where the Sensitive Attribute *is* the answer, so a Model
    with no access to it falls back on the majority class and the selection rates
    genuinely diverge. The message has to be checkable by reading it: the gap, the
    two groups and the numbers behind them."""
    run_id = run_id_of(client, unfair_version)
    body = report(client, run_id, sensitive_attribute="region", gap_threshold=0.0)

    raised = [c for c in body["checks_raised"] if c["kind"] == FAIRNESS_GAP_CHECK_KIND]
    assert raised, body["gaps_exceeding_threshold"]
    for check in raised:
        assert check["severity"] == "warning"
        assert check["subject_type"] == "training_run"
        assert check["subject_id"] == run_id
        assert check["step"] == "fairness"
        assert check["acknowledged"] is False
        details = check["details"]
        assert details["model"] == body["model"]
        assert details["sensitive_attribute"] == "region"
        assert details["gap"] in body["gaps_exceeding_threshold"]
        assert details["value"] > 0.0
        assert details["highest_group"] in {g["group"] for g in body["groups"]}
        assert details["lowest_group"] in {g["group"] for g in body["groups"]}
        # The message names the groups and the numbers, not just "a gap exists".
        assert details["highest_group"] in check["message"]
        assert details["lowest_group"] in check["message"]
        assert str(details["value"])[:5] in check["message"] or f"{details['value']:.3f}" in check[
            "message"
        ]
        assert str(details["threshold"]) in check["message"]
    # A warning needs an Acknowledgement, and it shows up on the run.
    assert body["unacknowledged_warnings"]
    read_back = client.get(f"/api/train/runs/{run_id}").json()
    kinds = {check["kind"] for check in read_back["checks"]}
    assert FAIRNESS_GAP_CHECK_KIND in kinds
    listed = client.get(
        f"/api/checks?subject_type=training_run&subject_id={run_id}"
    ).json()
    assert any(c["kind"] == FAIRNESS_GAP_CHECK_KIND for c in listed["checks"])


def test_the_widest_gap_says_it_is_the_widest(
    client: TestClient, unfair_version: str
) -> None:
    run_id = run_id_of(client, unfair_version)
    body = report(client, run_id, sensitive_attribute="region", gap_threshold=0.0)
    gaps = [c for c in body["checks_raised"] if c["kind"] == FAIRNESS_GAP_CHECK_KIND]
    assert gaps, "the unfair frame is what makes the widest-gap claim testable"
    widest_name = gaps[0]["details"]["largest_gap"]
    assert gaps[0]["details"]["gap"] == widest_name
    assert gaps[0]["details"]["largest_gap_value"] == body["largest_comparable_gap"]["value"]
    assert "widest" in gaps[0]["message"]
    # Every gap is named in every Check's details, so the reader can see the set.
    for check in gaps:
        assert set(check["details"]["gaps_exceeding_threshold"]) == {
            c["details"]["gap"] for c in gaps
        }


def test_a_repeat_request_does_not_pile_up_duplicate_checks(
    client: TestClient, unfair_version: str
) -> None:
    run_id = run_id_of(client, unfair_version)
    first = report(client, run_id, sensitive_attribute="region", gap_threshold=0.0)
    assert first["checks_raised"]
    second = report(client, run_id, sensitive_attribute="region", gap_threshold=0.0)
    assert second["checks_raised"] == [], "the same gaps were already reported"
    listed = client.get(f"/api/checks?subject_type=training_run&subject_id={run_id}").json()
    gap_checks = [c for c in listed["checks"] if c["kind"] == FAIRNESS_GAP_CHECK_KIND]
    assert len(gap_checks) == len(
        [c for c in first["checks_raised"] if c["kind"] == FAIRNESS_GAP_CHECK_KIND]
    )


def test_a_different_attribute_gets_its_own_checks(
    client: TestClient, unfair_version: str
) -> None:
    """`has_check` is subject-wide, so dedup has to be keyed by attribute too."""
    run_id = run_id_of(client, unfair_version)

    def keys(body: dict[str, Any]) -> set[tuple[str, str]]:
        return {
            (check["details"]["gap"], check["details"]["sensitive_attribute"])
            for check in body["checks_raised"]
            if check["kind"] == FAIRNESS_GAP_CHECK_KIND
        }

    first = report(client, run_id, sensitive_attribute="region", gap_threshold=0.0)
    second = report(client, run_id, sensitive_attribute="age", gap_threshold=0.0)
    first_keys, second_keys = keys(first), keys(second)
    assert first_keys and second_keys, "the second attribute's gaps were not silently swallowed"
    assert all(attribute == "age" for _gap, attribute in second_keys)
    assert first_keys.isdisjoint(second_keys)


def test_raise_check_false_measures_without_raising_anything(
    client: TestClient, unfair_version: str
) -> None:
    run_id = run_id_of(client, unfair_version)
    body = report(
        client,
        run_id,
        sensitive_attribute="region",
        gap_threshold=0.0,
        raise_check=False,
    )
    assert body["gaps_exceeding_threshold"], "the report itself still finds the gaps"
    assert body["checks_raised"] == []
    assert all(c["kind"] != FAIRNESS_GAP_CHECK_KIND for c in body["unacknowledged_warnings"])


def test_no_check_is_raised_by_a_training_run_that_was_never_asked(
    client: TestClient, version: str
) -> None:
    """The 'on demand' half: a Training Run computes no Fairness Report and
    registers no fairness Check of its own."""
    run = train(
        client,
        version_id=version,
        target="is_spam",
        models=["knn"],
        sensitive_attribute="region",
    )
    run_id = run["training_run_id"]
    # Nothing in the persisted run is a Fairness Report: no per-group metrics, no
    # gaps, and the Sensitive Attribute appears only where #31 put it.
    payload = json.dumps(run)
    for absent in ("gaps_exceeding_threshold", "demographic_parity", "largest_gap", "groups"):
        assert absent not in payload, absent
    listed = client.get(
        f"/api/checks?subject_type=training_run&subject_id={run_id}"
    ).json()
    assert all(c["kind"] != FAIRNESS_GAP_CHECK_KIND for c in listed["checks"])
    assert all(c["kind"] != FAIRNESS_USED_CHECK_KIND for c in listed["checks"])
    # And the run did record the attribute it was told about, so a later report
    # knows which one was declared.
    assert run["setup"]["sensitive_attribute"] == "region"


def test_registering_the_same_report_twice_is_a_no_op(
    client: TestClient, unfair_version: str
) -> None:
    """The dedup is in `register_fairness_checks` itself, not only in the route."""
    run_id = run_id_of(client, unfair_version)
    body = report(
        client, run_id, sensitive_attribute="region", gap_threshold=0.0, raise_check=False
    )
    store: Any = client.app.state.checks
    first = register_fairness_checks(store, body)
    second = register_fairness_checks(store, body)
    assert first, body["gaps_exceeding_threshold"]
    assert second == []


def test_the_threshold_comes_from_settings_and_the_request_overrides_it(
    client: TestClient, version: str
) -> None:
    run_id = run_id_of(client, version)
    settings = client.app.state.settings
    configured = report(client, run_id, sensitive_attribute="region")
    assert configured["threshold"] == DEFAULT_GAP_THRESHOLD
    assert configured["threshold_source"] == "settings"

    settings.save({"fairness_gap_threshold": 0.5})
    from_settings = report(client, run_id, sensitive_attribute="region")
    assert from_settings["threshold"] == 0.5
    assert from_settings["threshold_source"] == "settings"
    for gap in from_settings["gaps"]:
        assert gap["threshold"] == 0.5

    overridden = report(
        client, run_id, sensitive_attribute="region", gap_threshold=0.01
    )
    assert overridden["threshold"] == 0.01
    assert overridden["threshold_source"] == "request"
    # The settings value is untouched by a per-request override.
    assert settings.load()["fairness_gap_threshold"] == 0.5
    assert report(client, run_id, sensitive_attribute="region")["threshold"] == 0.5


# -- the encoding between the group column and the Target --------------------


def test_encoding_against_the_models_classes_refuses_an_unseen_class() -> None:
    classes = ["bool:False", "bool:True"]
    assert encode_with_classes([True, False, True], classes).tolist() == [1, 0, 1]
    with pytest.raises(FairnessError, match="not one of the Model's classes"):
        encode_with_classes([None], classes)


def test_the_positive_class_convention_matches_the_leaderboards() -> None:
    # The same convention the ROC-AUC scorer uses: the later class is positive.
    assert positive_class_code(["bool:False", "bool:True"]) == (1, "bool:True")
    assert positive_class_code([value_key(0), value_key(1)]) == (1, value_key(1))
