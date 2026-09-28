"""The Generation run API: preview, estimate, job → Version + checks."""

from __future__ import annotations

import time

import numpy as np
import pandas as pd
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from dat_distiller.generate.run import GenerationRequest, effective_seed, estimate
from dat_distiller.jev import FAKE_JEV_ENV  # noqa: F401  (env hygiene)
from dat_distiller.providers.registry import FAKE_PROVIDERS_ENV


@pytest.fixture
def project(client: TestClient) -> str:
    return client.post("/api/projects", json={"name": "gen"}).json()["id"]


@pytest.fixture
def sample_version(client: TestClient, project: str) -> str:
    rng = np.random.default_rng(3)
    df = pd.DataFrame(
        {
            "age": rng.integers(20, 70, 200),
            "plan": rng.choice(["basic", "pro"], 200),
        }
    )
    body = {
        "columns": list(df.columns),
        "rows": df.values.tolist(),
    }
    response = client.post(f"/api/projects/{project}/dataset_versions", json=body)
    assert response.status_code in (200, 201)
    return response.json()["id"]


@pytest.fixture
def fake_providers(monkeypatch) -> None:
    monkeypatch.setenv(FAKE_PROVIDERS_ENV, "1")
    monkeypatch.delenv(FAKE_JEV_ENV, raising=False)


def wait_for_job(client: TestClient, job_id: str, timeout: float = 15.0) -> dict:
    deadline = time.time() + timeout
    while time.time() < deadline:
        job = client.get(f"/api/jobs/{job_id}").json()
        if job["status"] in ("completed", "failed", "cancelled"):
            return job
        time.sleep(0.05)
    raise AssertionError(f"job {job_id} did not finish")


def test_estimate_endpoint(client: TestClient, sample_version: str, project: str) -> None:
    stats = client.post(
        "/api/generate/estimate",
        json={
            "project_id": project,
            "description": "customers",
            "mode": "statistical",
            "count": 100,
            "sample_version_id": sample_version,
        },
    ).json()
    assert stats["uses_provider"] is False
    assert stats["seed_set"] is False

    llm = client.post(
        "/api/generate/estimate",
        json={
            "project_id": project,
            "description": "customers",
            "mode": "llm",
            "count": 100,
            "sample_version_id": sample_version,
        },
    ).json()
    assert llm["estimated_provider_calls"] == 4  # 100 / batch of 25

    spec_only = client.post(
        "/api/generate/estimate",
        json={
            "project_id": project,
            "description": "customers",
            "mode": "statistical",
            "count": 10,
            "specs": [{"name": "x", "type": "number"}],
        },
    ).json()
    assert spec_only["seed_set"] is True and spec_only["uses_provider"] is True


def test_estimate_requires_source(project: str) -> None:
    with pytest.raises(ValueError):
        GenerationRequest(project_id=project, description="x y z", count=5)


def test_unseeded_request_is_reproducible(project: str) -> None:
    a = GenerationRequest(
        project_id=project, description="desc", specs=[{"name": "x", "type": "number"}], count=10
    )
    b = GenerationRequest(
        project_id=project, description="desc", specs=[{"name": "x", "type": "number"}], count=10
    )
    assert effective_seed(a) == effective_seed(b)
    assert effective_seed(GenerationRequest.model_validate({**a.model_dump(), "seed": 9})) == 9


def test_preview_every_mode(client: TestClient, project: str, sample_version: str, fake_providers) -> None:
    for mode in ("statistical", "llm", "hybrid"):
        response = client.post(
            "/api/generate/preview",
            json={
                "project_id": project,
                "description": "customers like these",
                "mode": mode,
                "count": 50,
                "sample_version_id": sample_version,
                "preview_rows": 6,
            },
        )
        assert response.status_code == 200, (mode, response.text)
        rows = response.json()["rows"]
        assert len(rows) == 6, mode
        assert set(rows[0]) == {"age", "plan"}


def test_run_job_produces_synthetic_version(client: TestClient, project: str, sample_version: str, fake_providers) -> None:
    start = client.post(
        "/api/generate/run",
        json={
            "project_id": project,
            "description": "more customers",
            "mode": "statistical",
            "count": 40,
            "sample_version_id": sample_version,
            "seed": 123,
        },
    )
    assert start.status_code == 202
    job = wait_for_job(client, start.json()["id"])
    assert job["status"] == "completed", job["error"]
    result = job["result"]
    assert result["rows"] == 40
    assert result["seed"] == 123

    version = client.get(f"/api/dataset-versions/{result['version_id']}").json()
    assert version["origin"] == "generated"
    assert version["seed"] == 123
    assert version["parent_id"] == sample_version
    assert version["provenance_summary"].get("synthetic") == 40
    assert version["meta"]["fidelity"]["generated_rows"] == 40

    preview = client.get(f"/api/dataset-versions/{result['version_id']}/preview").json()
    assert preview["total_rows"] == 40

    # the Check list mentions nothing risky for a faithful statistical run
    checks = client.get(f"/api/checks?subject_type=project&subject_id={project}").json()["checks"]
    assert "generation_soft_limit" not in {c["kind"] for c in checks}


def test_run_unseeded_twice_reproduces(client: TestClient, project: str, sample_version: str, fake_providers) -> None:
    seeds = []
    frames = []
    for _ in range(2):
        start = client.post(
            "/api/generate/run",
            json={
                "project_id": project,
                "description": "twice identical",
                "mode": "statistical",
                "count": 25,
                "sample_version_id": sample_version,
            },
        )
        job = wait_for_job(client, start.json()["id"])
        assert job["status"] == "completed"
        seeds.append(job["result"]["seed"])
        frames.append(
            client.get(f"/api/dataset-versions/{job['result']['version_id']}/download?format=jsonl").text
        )
    assert seeds[0] == seeds[1]
    assert frames[0] == frames[1]


def test_soft_limit_raises_check_not_error(client: TestClient, project: str, sample_version: str, fake_providers) -> None:
    client.app.state.settings.save({"soft_limits": {"generation_rows": 10}})
    start = client.post(
        "/api/generate/run",
        json={
            "project_id": project,
            "description": "over the limit",
            "mode": "statistical",
            "count": 20,
            "sample_version_id": sample_version,
        },
    )
    job = wait_for_job(client, start.json()["id"])
    assert job["status"] == "completed"  # a warning, not an error
    checks = client.get(f"/api/checks?subject_type=project&subject_id={project}").json()["checks"]
    limit_checks = [c for c in checks if c["kind"] == "generation_soft_limit"]
    assert limit_checks and limit_checks[0]["severity"] == "warning"


def test_run_requires_provider_only_when_needed(client: TestClient, project: str, sample_version: str, monkeypatch) -> None:
    monkeypatch.delenv(FAKE_PROVIDERS_ENV, raising=False)
    body = {
        "project_id": project,
        "description": "pure stats need no key",
        "mode": "statistical",
        "count": 5,
        "sample_version_id": sample_version,
    }
    start = client.post("/api/generate/run", json=body)
    assert start.status_code == 202  # no provider configured, still allowed
    assert wait_for_job(client, start.json()["id"])["status"] == "completed"

    llm = client.post(
        "/api/generate/run", json={**body, "mode": "llm", "description": "now we need one"}
    )
    assert llm.status_code == 422
    assert "key" in llm.text.lower() or "configur" in llm.text.lower()


def test_run_validates_balance_against_profile(client: TestClient, project: str, sample_version: str, fake_providers) -> None:
    start = client.post(
        "/api/generate/run",
        json={
            "project_id": project,
            "description": "bad balance column",
            "mode": "statistical",
            "count": 10,
            "sample_version_id": sample_version,
            "balance": {"nonexistent": {"x": 1.0}},
        },
    )
    assert start.status_code == 202
    job = wait_for_job(client, start.json()["id"])
    assert job["status"] == "failed"  # surfaces through the job, not the request
