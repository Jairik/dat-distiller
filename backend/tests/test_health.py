"""`GET /api/health` reports liveness, version, and the optional extras."""

from __future__ import annotations

import importlib.machinery
from typing import Any

import pytest
from fastapi.testclient import TestClient

from dat_distiller import __version__
from dat_distiller import extras as extras_module
from dat_distiller.api.app import create_app

EXTRAS = ("torch", "tensorflow", "presidio")


@pytest.fixture
def client() -> TestClient:
    with TestClient(create_app()) as test_client:
        yield test_client


def test_health_reports_ok_with_package_version(client: TestClient) -> None:
    response = client.get("/api/health")

    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ok"
    assert body["version"] == __version__


def test_health_lists_every_optional_extra_as_a_boolean(client: TestClient) -> None:
    extras = client.get("/api/health").json()["extras"]

    assert set(extras) == set(EXTRAS)
    assert all(isinstance(installed, bool) for installed in extras.values())


def test_extras_report_not_installed_when_imports_do_not_resolve(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    # The dev environment installs none of the heavy extras; assert the answer
    # rather than depending on what happens to be installed.
    monkeypatch.setattr(extras_module, "find_spec", lambda name: None)

    assert client.get("/api/health").json()["extras"] == dict.fromkeys(EXTRAS, False)


def test_an_extra_flips_to_installed_once_its_module_resolves(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    def fake_find_spec(name: str) -> Any:
        if name != "torch":
            return None
        return importlib.machinery.ModuleSpec(name, loader=None)

    monkeypatch.setattr(extras_module, "find_spec", fake_find_spec)

    assert client.get("/api/health").json()["extras"] == {
        "torch": True,
        "tensorflow": False,
        "presidio": False,
    }
