"""`serve` serves a built frontend at `/` with SPA fallback, and keeps API 404s JSON."""

from __future__ import annotations

from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from dat_distiller.api.app import create_app
from dat_distiller.api.frontend import FRONTEND_DIST_ENV
from dat_distiller.cli import DEFAULT_HOST, DEFAULT_PORT, build_parser


@pytest.fixture
def built_frontend(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """A stand-in for `frontend/dist`, pointed at via the env var."""
    dist = tmp_path / "dist"
    (dist / "assets").mkdir(parents=True)
    (dist / "index.html").write_text("<!doctype html><title>Dat Distiller</title>")
    (dist / "assets" / "index.js").write_text("console.log('distiller');")
    monkeypatch.setenv(FRONTEND_DIST_ENV, str(dist))
    return dist


@pytest.fixture
def client(built_frontend: Path) -> TestClient:
    with TestClient(create_app()) as test_client:
        yield test_client


def test_index_html_is_served_at_root(client: TestClient, built_frontend: Path) -> None:
    response = client.get("/")

    assert response.status_code == 200
    assert response.text == (built_frontend / "index.html").read_text()


def test_static_asset_is_served_from_the_build(client: TestClient) -> None:
    response = client.get("/assets/index.js")

    assert response.status_code == 200
    assert "text/javascript" in response.headers["content-type"]


def test_client_side_routes_fall_back_to_index_html(client: TestClient) -> None:
    for path in ("/projects", "/projects/7/train", "/settings"):
        response = client.get(path)

        assert response.status_code == 200, path
        assert "text/html" in response.headers["content-type"]
        assert response.text == "<!doctype html><title>Dat Distiller</title>"


def test_health_is_not_swallowed_by_the_fallback(client: TestClient) -> None:
    assert client.get("/api/health").json()["status"] == "ok"


def test_unknown_api_path_stays_a_json_404(client: TestClient) -> None:
    response = client.get("/api/does-not-exist")

    assert response.status_code == 404
    assert "json" in response.headers["content-type"]
    assert response.json() == {"detail": "Not Found"}


def test_without_a_build_nothing_is_served_at_root(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv(FRONTEND_DIST_ENV, str(tmp_path / "not-built-yet"))

    with TestClient(create_app()) as client:
        assert client.get("/projects").status_code == 404
        assert client.get("/api/health").json()["status"] == "ok"


def test_serve_defaults_to_localhost_on_the_app_port() -> None:
    args = build_parser().parse_args(["serve"])

    assert (args.host, args.port) == (DEFAULT_HOST, DEFAULT_PORT) == ("127.0.0.1", 8756)
    assert build_parser().parse_args(["serve", "--port", "9000"]).port == 9000
