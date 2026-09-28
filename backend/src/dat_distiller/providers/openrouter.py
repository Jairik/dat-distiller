"""The OpenRouter Provider: chat completions over HTTP, JSON-validated."""

from __future__ import annotations

from typing import Any

import httpx

from .base import (
    InvalidStructuredOutputError,
    ProviderNotConfiguredError,
    ProviderTimeoutError,
    parse_json_reply,
    validate_structured,
)

OPENROUTER_BASE = "https://openrouter.ai/api/v1"
DEFAULT_MODEL = "openai/gpt-4o-mini"


class OpenRouterProvider:
    id = "openrouter"

    def __init__(
        self,
        api_key: str,
        model: str | None = None,
        *,
        client: httpx.Client | None = None,
        timeout: float = 120.0,
        base_url: str = OPENROUTER_BASE,
    ) -> None:
        self.api_key = api_key
        self.model = model or DEFAULT_MODEL
        self.base_url = base_url
        self.timeout = timeout
        self._client = client or httpx.Client(timeout=timeout)

    def complete(self, prompt: str, schema: dict[str, Any]) -> dict[str, Any]:
        try:
            response = self._client.post(
                f"{self.base_url}/chat/completions",
                headers={
                    "Authorization": f"Bearer {self.api_key}",
                    "HTTP-Referer": "http://localhost",
                    "X-Title": "Dat Distiller",
                },
                json={
                    "model": self.model,
                    "messages": [{"role": "user", "content": prompt}],
                    # ask for JSON when the model supports it; we validate anyway
                    "response_format": {"type": "json_object"},
                },
            )
        except httpx.TimeoutException as exc:
            raise ProviderTimeoutError(f"OpenRouter timed out: {exc}") from exc
        if response.status_code == 401:
            raise ProviderNotConfiguredError("OpenRouter rejected the API key")
        if response.status_code != 200:
            raise InvalidStructuredOutputError(
                f"OpenRouter returned {response.status_code}: {response.text[:200]}"
            )
        try:
            content = response.json()["choices"][0]["message"]["content"]
        except (KeyError, IndexError, ValueError) as exc:
            raise InvalidStructuredOutputError(f"unexpected OpenRouter reply: {exc}") from exc
        payload = parse_json_reply(content)
        errors = validate_structured(payload, schema)
        if errors:
            raise InvalidStructuredOutputError("OpenRouter output failed schema validation", errors)
        return payload


def list_models(api_key: str, *, client: httpx.Client | None = None) -> list[dict[str, Any]]:
    """Model catalogue for the picker: id + friendly name."""
    client = client or httpx.Client(timeout=30.0)
    response = client.get(
        f"{OPENROUTER_BASE}/models", headers={"Authorization": f"Bearer {api_key}"}
    )
    response.raise_for_status()
    data = response.json().get("data", [])
    return [
        {"id": m["id"], "name": m.get("name", m["id"])}
        for m in data
        if isinstance(m, dict) and "id" in m
    ]
