"""Shared fixtures: every test gets an isolated data directory."""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from dat_distiller.api.app import create_app


@pytest.fixture(autouse=True)
def isolated_data_dir(tmp_path, monkeypatch):
    """All storage goes to a per-test directory."""
    monkeypatch.setenv("DAT_DISTILLER_DATA_DIR", str(tmp_path / "data"))
    return tmp_path / "data"


@pytest.fixture
def app(isolated_data_dir):
    return create_app()


@pytest.fixture
def client(app):
    with TestClient(app) as test_client:
        yield test_client
