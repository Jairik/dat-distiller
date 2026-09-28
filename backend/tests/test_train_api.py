"""Training setup endpoints: estimate, validate, run job, read the plan back."""

from __future__ import annotations

import json
import time

import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient

from dat_distiller.store.provenance import PROVENANCE_COLUMN
from dat_distiller.store.store import DatasetStore


def labeled_frame(rows: int = 20) -> pd.DataFrame:
    rng = np.random.default_rng(0)
    frame = pd.DataFrame(
        {
            "age": rng.integers(18, 80, rows),
            "spend": rng.normal(0, 1, rows).round(3),
            "region": rng.choice(["north", "south", "east"], rows),
            "bio": [f"row {i} alpha beta" for i in range(rows)],
        }
    )
    # not a function of the row's parity, so dropping either half keeps both classes
    frame["is_spam"] = pd.array([i % 3 == 0 for i in range(rows)], dtype="boolean")
    # half the Jev labels sit below the review threshold of 0.8
    frame["is_spam__confidence"] = pd.array(
        [0.4 if i % 2 == 0 else 0.95 for i in range(rows)], dtype="Float64"
    )
    frame[PROVENANCE_COLUMN] = [
        json.dumps(
            {
                "row_origin": "uploaded",
                "label_origins": {
                    "is_spam": {"origin": "jev", "confidence": 0.4 if i % 2 == 0 else 0.95}
                },
            }
        )
        for i in range(rows)
    ]
    return frame


@pytest.fixture
def version(client: TestClient) -> str:
    project_id = client.post("/api/projects", json={"name": "train lab"}).json()["id"]
    store: DatasetStore = client.app.state.store
    return store.create_version(project_id, labeled_frame(20), origin="uploaded").id


def wait_for_job(client: TestClient, job_id: str, timeout: float = 15.0) -> dict:
    deadline = time.time() + timeout
    while time.time() < deadline:
        job = client.get(f"/api/jobs/{job_id}").json()
        if job["status"] in ("completed", "failed", "cancelled"):
            return job
        time.sleep(0.05)
    raise AssertionError("job did not finish")


def test_estimate_endpoint(client: TestClient, version: str) -> None:
    response = client.post(
        "/api/train/estimate", json={"version_id": version, "target": "is_spam"}
    )
    assert response.status_code == 200
    body = response.json()
    assert body["task_type"] == "classification"
    assert body["rows"] == 20
    assert body["train_rows"] == 16
    assert body["test_rows"] == 4
    assert body["test_size"] == 0.2
    assert body["stratified"] is True
    assert "is_spam__confidence" in body["excluded_columns"]
    assert body["excluded_columns"]["is_spam__confidence"] == "target_sibling"
    assert body["sensitive_attribute"] is None
    assert body["unreviewed"]["unreviewed_rows"] >= 0


def test_validate_endpoint_proposes_the_plan(client: TestClient, version: str) -> None:
    response = client.post(
        "/api/train/validate",
        json={
            "version_id": version,
            "target": "is_spam__confidence",
            "sensitive_attribute": "region",
            "exclude_features": ["bio"],
        },
    )
    assert response.status_code == 200
    body = response.json()
    assert body["valid"] is True
    assert body["inferred_task_type"] == "regression"
    assert body["task_type"] == "regression"
    assert body["task_type_changed"] is False
    assert body["label_family"] == "is_spam"
    assert body["excluded_columns"]["is_spam"] == "target_sibling"
    assert body["excluded_columns"]["region"] == "sensitive_attribute"
    assert body["excluded_columns"]["bio"] == "user_excluded"
    assert "region" not in body["feature_columns"]
    assert "bio" not in body["feature_columns"]


def test_validate_endpoint_reports_a_changed_task_type(client: TestClient, version: str) -> None:
    body = client.post(
        "/api/train/validate",
        json={"version_id": version, "target": "spend", "task_type": "classification"},
    ).json()
    assert body["inferred_task_type"] == "regression"
    assert body["task_type"] == "classification"
    assert body["task_type_changed"] is True


def test_setup_endpoint_runs_a_job_and_persists_the_plan(client: TestClient, version: str) -> None:
    start = client.post(
        "/api/train/setup", json={"version_id": version, "target": "is_spam", "sensitive_attribute": "region"}
    )
    assert start.status_code == 202
    job = wait_for_job(client, start.json()["id"])
    assert job["status"] == "completed", job["error"]
    result = job["result"]
    assert result["version_id"] == version
    assert result["task_type"] == "classification"
    assert result["train_rows"] == 16
    assert "region" not in result["feature_columns"]
    assert "label_sibling_exclusion" in result["checks_raised"]

    plan = client.get(f"/api/train/setup/{job['id']}")
    assert plan.status_code == 200
    body = plan.json()
    assert body["training_run_id"] == job["id"]
    assert body["setup"]["target"] == "is_spam"
    assert body["preprocessing"]["features"] == result["feature_columns"]
    assert len(body["split_indices"]["train"]) == 16
    assert len(body["split_indices"]["test"]) == 4
    kinds = {check["kind"] for check in body["checks"]}
    assert "label_sibling_exclusion" in kinds
    assert all(check["subject_type"] == "training_run" for check in body["checks"])


def test_setup_endpoint_raises_a_class_imbalance_check(client: TestClient) -> None:
    project_id = client.post("/api/projects", json={"name": "skewed"}).json()["id"]
    frame = labeled_frame(30)
    frame["region"] = ["north"] * 28 + ["south", "east"]
    store: DatasetStore = client.app.state.store
    skewed = store.create_version(project_id, frame, origin="uploaded").id

    start = client.post(
        "/api/train/setup", json={"version_id": skewed, "target": "region", "features": ["age"]}
    )
    job = wait_for_job(client, start.json()["id"])
    assert job["status"] == "completed", job["error"]
    assert "class_imbalance" in job["result"]["checks_raised"]

    body = client.get(f"/api/train/setup/{job['id']}").json()
    checks = {c["kind"]: c for c in body["checks"]}
    assert checks["class_imbalance"]["severity"] == "warning"
    assert checks["class_imbalance"]["details"]["target"] == "region"
    assert body["unacknowledged_warnings"], "warnings need an Acknowledgement before continuing"

    listed = client.get(f"/api/checks?subject_type=training_run&subject_id={job['id']}").json()
    assert any(c["kind"] == "class_imbalance" for c in listed["checks"])


def test_setup_endpoint_warns_about_unreviewed_labels(client: TestClient, version: str) -> None:
    start = client.post(
        "/api/train/setup", json={"version_id": version, "target": "is_spam", "features": ["age"]}
    )
    job = wait_for_job(client, start.json()["id"])
    assert job["status"] == "completed", job["error"]
    assert job["result"]["unreviewed_rows"] == 10
    body = client.get(f"/api/train/setup/{job['id']}").json()
    unreviewed = next(c for c in body["checks"] if c["kind"] == "unreviewed_labels")
    assert unreviewed["severity"] == "warning"
    assert unreviewed["details"]["confidence_column"] == "is_spam__confidence"


def test_setup_endpoint_excludes_unreviewed_rows_when_asked(client: TestClient) -> None:
    project_id = client.post("/api/projects", json={"name": "unreviewed"}).json()["id"]
    store: DatasetStore = client.app.state.store
    version = store.create_version(project_id, labeled_frame(20), origin="uploaded").id
    start = client.post(
        "/api/train/setup",
        json={"version_id": version, "target": "is_spam", "features": ["age"], "exclude_unreviewed": True},
    )
    assert start.status_code == 202
    job = wait_for_job(client, start.json()["id"])
    assert job["status"] == "completed", job["error"]
    result = job["result"]
    assert result["unreviewed_rows"] == 0
    assert result["dropped_unreviewed"] == 10
    assert result["train_rows"] + result["test_rows"] == 10
    body = client.get(f"/api/train/setup/{job['id']}").json()
    unreviewed = next(c for c in body["checks"] if c["kind"] == "unreviewed_labels")
    assert unreviewed["severity"] == "info"


def test_setup_endpoint_validates_the_request(client: TestClient, version: str) -> None:
    missing_version = client.post("/api/train/setup", json={"version_id": "nope", "target": "age"})
    assert missing_version.status_code == 404
    missing_target = client.post("/api/train/setup", json={"version_id": version, "target": "nope"})
    assert missing_target.status_code == 422
    bad_task_type = client.post(
        "/api/train/setup", json={"version_id": version, "target": "age", "task_type": "clustering"}
    )
    assert bad_task_type.status_code == 422
    no_features = client.post(
        "/api/train/validate", json={"version_id": version, "target": "age", "features": ["age"]}
    )
    assert no_features.status_code == 422


def test_reading_an_unknown_training_run(client: TestClient) -> None:
    assert client.get("/api/train/setup/nope").status_code == 404
