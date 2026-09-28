"""What the app does when the optional extras are not installed.

Every other training test is guarded with ``importorskip``, so this module is
the one that has to run on a bare install. It does not import scikit-learn or
LightGBM at all: it makes the registry *believe* they are missing (which is
exactly the state a bare install is in) and checks that the answer is a
readable refusal or a reduced Model list — never a crash, never a silently
wrong Model.

The two guards in here that do need the real libraries live in
``test_trainers.py`` and ``test_train_run_api.py``.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient

from dat_distiller.extras import EXTRA_MODULES
from dat_distiller.store.store import DatasetStore
from dat_distiller.training import trainers
from dat_distiller.training.trainers import ModelUnavailableError, get_spec


@pytest.fixture
def project_id(client: TestClient) -> str:
    return client.post("/api/projects", json={"name": "bare install"}).json()["id"]


@pytest.fixture
def version(client: TestClient, project_id: str) -> str:
    rng = np.random.default_rng(5)
    rows = 40
    frame = pd.DataFrame(
        {
            "age": rng.integers(18, 80, rows),
            "spend": rng.normal(0, 1, rows).round(3),
            "is_spam": pd.array(rng.integers(0, 2, rows).astype(bool), dtype="boolean"),
        }
    )
    store: DatasetStore = client.app.state.store
    return store.create_version(project_id, frame, origin="uploaded").id


@pytest.fixture
def no_extras(monkeypatch: pytest.MonkeyPatch) -> pytest.MonkeyPatch:
    """Pretend this install has neither scikit-learn nor LightGBM."""
    real = trainers.is_installed

    def is_installed(module: str) -> bool:
        return False if module in ("sklearn", "lightgbm") else real(module)

    monkeypatch.setattr(trainers, "is_installed", is_installed)
    return monkeypatch


# -- the health and registry endpoints --------------------------------------


def test_health_lists_the_trainer_extras_as_booleans(client: TestClient) -> None:
    extras = client.get("/api/health").json()["extras"]
    assert "sklearn" in extras and "lightgbm" in extras
    assert isinstance(extras["sklearn"], bool)
    assert isinstance(extras["lightgbm"], bool)
    assert set(EXTRA_MODULES) == {"torch", "tensorflow", "presidio", "sklearn", "lightgbm"}


def test_the_registry_always_lists_every_model(client: TestClient) -> None:
    body = client.get("/api/train/models").json()
    assert len(body["models"]) == len(trainers.MODEL_SPECS)
    for model in body["models"]:
        assert model["available"] == trainers.is_available(get_spec(model["name"]))
        assert model["task_types"]


def test_the_registry_reports_the_models_this_install_cannot_run(
    client: TestClient, no_extras: pytest.MonkeyPatch
) -> None:
    body = client.get("/api/train/models").json()
    by_name = {model["name"]: model for model in body["models"]}
    assert by_name["logistic_regression"]["available"] is False
    assert by_name["lightgbm"]["available"] is False
    # The metadata is still there, so the UI can say what would be gained.
    assert by_name["lightgbm"]["library"] == "lightgbm"
    assert by_name["lightgbm"]["search_space"]


# -- refusals ---------------------------------------------------------------


def test_selecting_a_model_whose_extra_is_missing_is_refused(
    client: TestClient, version: str, no_extras: pytest.MonkeyPatch
) -> None:
    response = client.post(
        "/api/train/plan", json={"version_id": version, "target": "is_spam", "models": ["knn"]}
    )
    assert response.status_code == 422
    detail = response.json()["detail"]
    assert "knn" in detail and "sklearn" in detail
    assert "--extra sklearn" in detail, "the error should say how to get the Model"

    lightgbm = client.post(
        "/api/train/plan",
        json={"version_id": version, "target": "is_spam", "models": ["lightgbm"]},
    )
    assert lightgbm.status_code == 422
    assert "lightgbm" in lightgbm.json()["detail"]


def test_the_run_endpoint_refuses_before_it_starts_a_job(
    client: TestClient, version: str, no_extras: pytest.MonkeyPatch
) -> None:
    response = client.post(
        "/api/train/run", json={"version_id": version, "target": "is_spam", "models": ["knn"]}
    )
    assert response.status_code == 422
    assert "sklearn" in response.json()["detail"]


def test_a_default_selection_falls_back_to_the_installed_models(
    client: TestClient, version: str, no_extras: pytest.MonkeyPatch
) -> None:
    """With nothing installed the plan says so rather than starting a doomed job."""
    response = client.post("/api/train/plan", json={"version_id": version, "target": "is_spam"})
    assert response.status_code == 422
    assert "no Model is installed" in response.json()["detail"]
    assert "--extra sklearn" in response.json()["detail"]


def test_a_partial_install_keeps_the_models_it_has(
    client: TestClient, version: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Every extra but LightGBM present: the default selection drops it and says so.

    Patched wholesale rather than off the real install, so this test means the
    same thing on a bare install and on a full one.
    """
    monkeypatch.setattr(trainers, "is_installed", lambda module: module != "lightgbm")
    plan = client.post(
        "/api/train/plan", json={"version_id": version, "target": "is_spam"}
    ).json()
    assert "lightgbm" not in plan["model_names"]
    assert {"logistic_regression", "svm", "random_forest", "knn", "naive_bayes"} <= set(
        plan["model_names"]
    )
    assert plan["unavailable_models"] == [
        {"model": "lightgbm", "reason": "needs the optional extra 'lightgbm'"}
    ]


def test_building_a_model_without_its_extra_is_a_readable_error(no_extras: pytest.MonkeyPatch) -> None:
    for name in ("knn", "lightgbm", "random_forest"):
        with pytest.raises(ModelUnavailableError, match="not installed"):
            trainers.build_estimator(get_spec(name), get_spec(name).task_types[0])
