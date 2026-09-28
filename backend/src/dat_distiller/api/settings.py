"""Settings API: preferences, key status (never values), Provider detection."""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel, Field

from ..settings import CLI_PROVIDERS, KEY_ENV_VARS, SettingsStore

router = APIRouter(tags=["settings"])


class SettingsPatch(BaseModel):
    default_provider: str | None = None
    models: dict[str, str] | None = None
    soft_limits: dict[str, int] | None = None
    review_threshold: float | None = Field(default=None, ge=0, le=1)
    fairness_gap_threshold: float | None = Field(default=None, ge=0, le=1)


class KeyIn(BaseModel):
    key: str = Field(min_length=1, max_length=400)


def _settings(request: Request) -> SettingsStore:
    return request.app.state.settings


def _payload(request: Request) -> dict[str, Any]:
    store = _settings(request)
    settings = store.load()
    return {
        **settings,
        "keys": {provider: store.key_status(provider) for provider in KEY_ENV_VARS},
        "providers": [
            {
                "id": name,
                "kind": "cli",
                "available": store.detect_cli(name) is not None,
            }
            for name in CLI_PROVIDERS
        ]
        + [
            {
                "id": "openrouter",
                "kind": "http",
                "available": store.key_status("openrouter")["set"],
            }
        ],
    }


@router.get("/settings")
def get_settings(request: Request) -> dict[str, Any]:
    return _payload(request)


@router.put("/settings")
def update_settings(body: SettingsPatch, request: Request) -> dict[str, Any]:
    try:
        _settings(request).save(body.model_dump(exclude_unset=True))
    except ValueError as exc:
        raise HTTPException(422, str(exc)) from exc
    return _payload(request)


@router.put("/settings/keys/{provider}")
def set_key(provider: str, body: KeyIn, request: Request) -> dict[str, Any]:
    try:
        _settings(request).set_key(provider, body.key)
    except ValueError as exc:
        raise HTTPException(422, str(exc)) from exc
    return {"keys": {p: _settings(request).key_status(p) for p in KEY_ENV_VARS}}


@router.delete("/settings/keys/{provider}")
def clear_key(provider: str, request: Request) -> dict[str, Any]:
    try:
        _settings(request).clear_key(provider)
    except ValueError as exc:
        raise HTTPException(422, str(exc)) from exc
    return {"keys": {p: _settings(request).key_status(p) for p in KEY_ENV_VARS}}
