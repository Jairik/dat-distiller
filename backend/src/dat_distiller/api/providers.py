"""Provider endpoints: options, and the OpenRouter model list."""

from __future__ import annotations

import httpx
from fastapi import APIRouter, HTTPException, Request

from ..providers import provider_options
from ..providers.openrouter import list_models

router = APIRouter(tags=["providers"])


@router.get("/providers/options")
def providers_options(request: Request) -> dict:
    return {"providers": provider_options(request.app.state.settings)}


@router.get("/providers/openrouter/models")
def openrouter_models(request: Request) -> dict:
    key = request.app.state.settings.api_key("openrouter")
    if not key:
        raise HTTPException(422, "set an OpenRouter API key first (Settings)")
    try:
        models = list_models(key)
    except httpx.HTTPError as exc:
        raise HTTPException(502, f"could not list OpenRouter models: {exc}") from exc
    return {"models": models}
