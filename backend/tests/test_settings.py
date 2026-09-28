"""Settings: 0600 files, env-key priority, detection, and never leaking keys."""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from dat_distiller import settings as settings_module
from dat_distiller.settings import SettingsStore, file_mode


@pytest.fixture
def store(isolated_data_dir) -> SettingsStore:
    return SettingsStore(isolated_data_dir.parent / "config")


def test_config_files_are_0600(store: SettingsStore) -> None:
    store.save({"default_provider": "codex"})
    store.set_key("openrouter", "sk-or-secret")
    assert file_mode(store.settings_path) == 0o600
    assert file_mode(store.keys_path) == 0o600


def test_defaults_and_partial_updates(store: SettingsStore) -> None:
    loaded = store.load()
    assert loaded["soft_limits"] == {
        "upload_rows": 500_000,
        "generation_rows": 10_000,
        "labeling_calls": 5_000,
    }
    assert loaded["default_provider"] == "openrouter"
    store.save({"soft_limits": {"generation_rows": 50}})
    assert store.load()["soft_limits"] == {
        "upload_rows": 500_000,
        "generation_rows": 50,
        "labeling_calls": 5_000,
    }
    with pytest.raises(ValueError):
        store.save({"default_provider": "gpt-master"})
    with pytest.raises(ValueError):
        store.save({"nonsense": 1})


def test_environment_variable_wins_and_reports_source(
    store: SettingsStore, monkeypatch
) -> None:
    store.set_key("typesafe", "file-key")
    assert store.key_status("typesafe") == {"set": True, "source": "file"}
    assert store.api_key("typesafe") == "file-key"

    monkeypatch.setenv("TYPESAFE_API_KEY", "env-key")
    assert store.api_key("typesafe") == "env-key"  # env beats the file
    assert store.key_status("typesafe") == {"set": True, "source": "env"}

    monkeypatch.delenv("TYPESAFE_API_KEY")
    store.clear_key("typesafe")
    assert store.api_key("typesafe") is None
    assert store.key_status("typesafe") == {"set": False, "source": None}


def test_provider_detection(store: SettingsStore, monkeypatch) -> None:
    monkeypatch.setattr(settings_module.shutil, "which", lambda name: "/usr/bin/fake" if name == "claude" else None)
    assert store.detect_cli("claude") == "/usr/bin/fake"
    assert store.detect_cli("codex") is None


# -- API ----------------------------------------------------------------------


def test_api_reports_status_never_key_values(client: TestClient) -> None:
    response = client.put("/api/settings/keys/openrouter", json={"key": "sk-super-secret-42"})
    assert response.status_code == 200
    assert response.json()["keys"]["openrouter"] == {"set": True, "source": "file"}

    for url in ("/api/settings", f"/api/health"):
        body = client.get(url).text
        assert "sk-super-secret-42" not in body, url

    payload = client.get("/api/settings").json()
    assert payload["keys"]["openrouter"] == {"set": True, "source": "file"}
    assert payload["keys"]["typesafe"] == {"set": False, "source": None}

    assert client.delete("/api/settings/keys/openrouter").json()["keys"]["openrouter"] == {
        "set": False,
        "source": None,
    }
    assert client.put("/api/settings/keys/typesafe", json={"key": "k"}).status_code == 200
    assert client.put("/api/settings/keys/whatsapp", json={"key": "k"}).status_code == 422


def test_env_key_shows_env_source_through_the_api(client: TestClient, monkeypatch) -> None:
    monkeypatch.setenv("OPENROUTER_API_KEY", "env-secret")
    payload = client.get("/api/settings").json()
    assert payload["keys"]["openrouter"] == {"set": True, "source": "env"}
    assert "env-secret" not in client.get("/api/settings").text


def test_settings_update_and_provider_flags(client: TestClient, monkeypatch) -> None:
    monkeypatch.setattr(settings_module.shutil, "which", lambda name: None)
    updated = client.put(
        "/api/settings",
        json={
            "default_provider": "claude",
            "models": {"claude": "agent default"},
            "review_threshold": 0.65,
        },
    ).json()
    assert updated["default_provider"] == "claude"
    assert updated["models"]["claude"] == "agent default"
    assert updated["review_threshold"] == 0.65
    by_id = {p["id"]: p for p in updated["providers"]}
    assert by_id["claude"]["available"] is False  # no CLIs on PATH in CI
    assert by_id["openrouter"]["available"] is False  # no key set

    assert client.put("/api/settings", json={"default_provider": "x"}).status_code == 422
    assert client.put("/api/settings", json={"review_threshold": 7}).status_code == 422
