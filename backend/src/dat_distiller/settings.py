"""Settings: keys, Provider defaults, soft limits, thresholds.

Stored under a config directory (default ``~/.config/dat-distiller/``,
overridable with ``DAT_DISTILLER_CONFIG_DIR`` for tests):

- ``settings.json``  — non-secret preferences, file mode 0600
- ``keys.json``      — API keys, file mode 0600, separate so they are never
  swept into another payload by accident

Environment variables ``OPENROUTER_API_KEY`` and ``TYPESAFE_API_KEY`` take
priority over stored keys. The API reports whether a key is set and its
source (``env`` or ``file``) but NEVER the key itself.
"""

from __future__ import annotations

import json
import os
import shutil
import stat
from pathlib import Path
from typing import Any

CONFIG_DIR_ENV = "DAT_DISTILLER_CONFIG_DIR"

KEY_ENV_VARS = {"openrouter": "OPENROUTER_API_KEY", "typesafe": "TYPESAFE_API_KEY"}
CLI_PROVIDERS = ("claude", "codex", "opencode")
PROVIDERS = CLI_PROVIDERS + ("openrouter",)

DEFAULT_SETTINGS: dict[str, Any] = {
    "default_provider": "openrouter",
    "models": {},  # provider id -> model id (or "agent default" style id)
    "soft_limits": {
        "upload_rows": 500_000,
        "generation_rows": 10_000,
        "labeling_calls": 5_000,
    },
    "review_threshold": 0.8,  # below this, Jev labels go to the Review Queue
    "fairness_gap_threshold": 0.1,
}


def config_dir() -> Path:
    raw = os.environ.get(CONFIG_DIR_ENV)
    root = Path(raw) if raw else Path.home() / ".config" / "dat-distiller"
    root.mkdir(parents=True, exist_ok=True)
    return root


def _read_json(path: Path, default: dict) -> dict:
    if not path.exists():
        return dict(default)
    return json.loads(path.read_text())


def _write_json_0600(path: Path, payload: dict) -> None:
    # Create with 0600 from the start (no world-readable window).
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    with os.fdopen(fd, "w") as handle:
        json.dump(payload, handle, indent=2)


class SettingsStore:
    """Loads/saves user settings and resolves API keys with env priority."""

    def __init__(self, directory: Path | None = None) -> None:
        self.dir = directory or config_dir()
        self.dir.mkdir(parents=True, exist_ok=True)

    # -- files ---------------------------------------------------------------

    @property
    def settings_path(self) -> Path:
        return self.dir / "settings.json"

    @property
    def keys_path(self) -> Path:
        return self.dir / "keys.json"

    def load(self) -> dict[str, Any]:
        settings = dict(DEFAULT_SETTINGS)
        stored = _read_json(self.settings_path, {})
        for key, value in stored.items():
            if key == "soft_limits":
                merged = dict(DEFAULT_SETTINGS["soft_limits"])
                merged.update(value)
                settings["soft_limits"] = merged
            else:
                settings[key] = value
        return settings

    def save(self, updates: dict[str, Any]) -> dict[str, Any]:
        settings = self.load()
        for key, value in updates.items():
            if value is None:
                continue
            if key == "default_provider" and value not in PROVIDERS:
                raise ValueError(f"unknown provider {value!r}")
            if key == "soft_limits":
                known = set(DEFAULT_SETTINGS["soft_limits"])
                unknown = set(value) - known
                if unknown:
                    raise ValueError(f"unknown soft limits: {sorted(unknown)}")
                merged = dict(settings["soft_limits"])
                merged.update(value)
                settings["soft_limits"] = merged
            elif key == "models":
                settings["models"] = {**settings["models"], **value}
            elif key in DEFAULT_SETTINGS:
                settings[key] = value
            else:
                raise ValueError(f"unknown setting {key!r}")
        _write_json_0600(self.settings_path, settings)
        return settings

    # -- API keys --------------------------------------------------------------

    def stored_keys(self) -> dict[str, str]:
        return _read_json(self.keys_path, {})

    def set_key(self, provider: str, value: str) -> None:
        if provider not in KEY_ENV_VARS:
            raise ValueError(f"no key slot for {provider!r}")
        keys = self.stored_keys()
        keys[provider] = value
        _write_json_0600(self.keys_path, keys)

    def clear_key(self, provider: str) -> None:
        if provider not in KEY_ENV_VARS:
            raise ValueError(f"no key slot for {provider!r}")
        keys = self.stored_keys()
        keys.pop(provider, None)
        _write_json_0600(self.keys_path, keys)

    def api_key(self, provider: str) -> str | None:
        """Effective key: environment variable wins over the stored file."""
        env_value = os.environ.get(KEY_ENV_VARS.get(provider, ""))
        if env_value:
            return env_value
        return self.stored_keys().get(provider)

    def key_status(self, provider: str) -> dict[str, Any]:
        """What the API may say about a key: set + source, never the value."""
        if os.environ.get(KEY_ENV_VARS.get(provider, "")):
            return {"set": True, "source": "env"}
        if provider in self.stored_keys():
            return {"set": True, "source": "file"}
        return {"set": False, "source": None}

    # -- Provider detection ------------------------------------------------------

    def detect_cli(self, name: str) -> str | None:
        return shutil.which(name)


def file_mode(path: Path) -> int:
    return stat.S_IMODE(path.stat().st_mode)
